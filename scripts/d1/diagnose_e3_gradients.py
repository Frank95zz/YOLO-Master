"""Read-only reconstruction of the E3 epoch-boundary gradient probe."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.d1 import run_e3
from scripts.d1.p1p2_runtime import restore_rank_buffers, restore_rng
from ultralytics.cfg import get_cfg
from ultralytics.data.d1_cache import D1FeatureCacheDataset, move_d1_batch_to_device
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="b0.1-z0.001-g0.1-s0")
    parser.add_argument("--softmax-fp64", action="store_true")
    parser.add_argument("--softmax-backward-fp64", action="store_true")
    parser.add_argument("--repair-dir", type=Path)
    args = parser.parse_args()
    global e3_mechanism
    if args.repair_dir:
        for name in ("e3_probe_precision", "e3_mechanism"):
            module_name = "scripts.d1." + name
            module_spec = importlib.util.spec_from_file_location(module_name, args.repair_dir / f"{name}.py")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        e3_mechanism = module
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    init_seeds(1 + rank, deterministic=True)
    mat = run_e3.matrix(args.workspace)
    spec = run_e3.spec_for(mat, run_e3.resolve_candidate(args.candidate), "screen")
    state = torch.load(Path(spec["output"]) / "resume.pt", map_location="cpu", weights_only=False)
    if state["identity"] != spec["identity"]:
        raise ValueError("Resume identity mismatch")
    model = run_e3.construct_model(spec["candidate"]).to(device).train()
    model.load_state_dict(state["model"], strict=True)
    restore_rank_buffers(model, state["ranks"][rank]["model_buffers"])
    model.args = get_cfg(overrides=run_e3.overrides(mat, spec))
    model.criterion = model.init_criterion()
    native = model.criterion.native_criterion
    for key, value in state["criterion"].items():
        setattr(native, key, value)
    # Match the already-registered epoch-boundary resume policy.
    from ultralytics.nn.modules.routing_protocol import anneal_mixture_temperatures

    anneal_mixture_temperatures(
        model, factor=model.args.moa_mot_temperature_factor, min_temp=model.args.moa_mot_min_temperature
    )
    data = YAML.load(model.args.data)
    dataset = D1FeatureCacheDataset(
        img_path=data["train"],
        cache_dir=Path(mat["cache"]) / "visdrone-train",
        data=data,
        hyp=model.args,
        trusted_cache=True,
        max_open_shards=4,
        prefetch_factor=1,
    )
    indices = [2 * rank, 2 * rank + 1]
    batch = move_d1_batch_to_device(
        dataset.collate_fn([dataset[i] for i in indices]), device, feature_dtype=torch.float16
    )
    restore_rng(state["ranks"][rank]["rng"], device)
    args.output.mkdir(parents=True, exist_ok=True)
    before = {k: v.clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    captures = []

    def capture(native, auxiliary, parameters, parameter_names=None):
        parameters, names = tuple(parameters), tuple(parameter_names)
        detection = torch.autograd.grad(native, parameters, retain_graph=True, allow_unused=True)
        aux = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
        total = torch.autograd.grad(native + auxiliary, parameters, allow_unused=True)
        rows, gradients = [], {}
        for name, p, a, b, c in zip(names, parameters, detection, aux, total):
            a, b, c = (torch.zeros_like(p) if v is None else v for v in (a, b, c))
            if not all(torch.isfinite(v).all() for v in (a, b, c)):
                raise FloatingPointError(name)
            delta = ((a + b).double() - c.double()).abs()
            tolerance = 2e-5 + 2e-4 * c.double().abs()
            failed = delta > tolerance
            selected = failed.nonzero().tolist()
            detail = {
                "name": name,
                "failed": int(failed.sum()),
                "elements": c.numel(),
                "max_abs": float(delta.max()),
                "max_tolerance_ratio": float((delta / tolerance).max()),
                "indices": selected,
                "values": [
                    {"detection": float(a[tuple(i)]), "aux": float(b[tuple(i)]), "total": float(c[tuple(i)])}
                    for i in selected[:32]
                ],
                "detection_norm": float(a.norm()),
                "aux_norm": float(b.norm()),
                "total_norm": float(c.norm()),
            }
            rows.append(detail)
            gradients[name] = {"detection": a.cpu(), "aux": b.cpu(), "total": c.cpu()}
        captures.append(rows)
        torch.save(gradients, args.output / f"rank-{rank}-gradients-{len(captures)}.pt")
        return [{"detection": r["detection_norm"], "aux": r["aux_norm"], "total": r["total_norm"]} for r in rows]

    if args.softmax_fp64 or args.softmax_backward_fp64:
        original_constructor = e3_mechanism.D1FoundationDetectionModel

        class AccurateSoftmax(torch.autograd.Function):
            @staticmethod
            def forward(ctx, scaled):
                result = torch.softmax(scaled, dim=-1)
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, gradient):
                (probs,) = ctx.saved_tensors
                p, g = probs.double(), gradient.double()
                return (p * (g - (g * p).sum(dim=-1, keepdim=True))).to(gradient.dtype)

        def construct(*positional, **kwargs):
            copied = original_constructor(*positional, **kwargs)

            def stable_softmax(router, inputs, output):
                logits, _ = output
                scaled = logits / router.temperature.clamp_min(0.1)
                if args.softmax_backward_fp64:
                    return logits, AccurateSoftmax.apply(scaled)
                return logits, torch.softmax(scaled.double(), dim=-1).to(logits.dtype)

            for mixture in copied.mixtures.values():
                mixture.router.register_forward_hook(stable_softmax)
            return copied

        e3_mechanism.D1FoundationDetectionModel = construct
    strict_gradients = e3_mechanism.separated_gradients
    e3_mechanism.separated_gradients = capture
    reports = [e3_mechanism.probe(model, batch) for _ in range(2)]
    strict_passed = None
    if args.repair_dir:
        e3_mechanism.separated_gradients = strict_gradients
        e3_mechanism.probe(model, batch)
        strict_passed = True
    if not all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise ValueError("Diagnostic changed source tensors")
    result = {
        "rank": rank,
        "indices": indices,
        "resume_epoch": state["epoch"] + 1,
        "source_unchanged": True,
        "diagnostic_only": True,
        "strict_passed": strict_passed,
        "probe": reports,
        "gradients": captures,
    }
    (args.output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rank": rank, "failed": [[r for r in rows if r["failed"]] for rows in captures]}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
