"""Offline contract checks for the D1 total-system-matched RGB baseline."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml"
TEACHER_PARAMS = 21_596_544


def test_standard_topology_and_explicit_scale():
    cfg = YAML.load(CONFIG)
    standard = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    assert cfg["backbone"] == standard["backbone"]
    assert cfg["head"] == standard["head"]
    assert cfg["scale"] == "l"
    assert cfg["scales"] == {"l": [1.0, 0.9375, 512]}
    assert cfg["end2end"] is True and cfg["reg_max"] == 1


@pytest.mark.parametrize(
    ("nc", "expected_downstream", "expected_scratch"),
    [(10, 1_340_259, 23_032_340), (80, 1_404_839, 23_133_560)],
)
def test_total_match_forward_and_strict_reload(nc, expected_downstream, expected_scratch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        frozen_cfg = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml")
        frozen_cfg["detect"]["nc"] = nc
        downstream = D1FoundationDetectionModel(frozen_cfg)
        assert sum(p.numel() for p in downstream.parameters()) == expected_downstream
        del downstream
        cfg = YAML.load(CONFIG)
        cfg["nc"] = nc
        model = DetectionModel(cfg, verbose=False)
        assert sum(p.numel() for p in model.parameters()) == expected_scratch
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_scratch
        assert abs(expected_scratch / (TEACHER_PARAMS + expected_downstream) - 1) < 0.01
        assert model.stride.tolist() == [8, 16, 32]
        export_limit = 500 if nc == 10 else 300
        model.model[-1].max_det = export_limit
        shapes = []
        handle = model.model[-1].register_forward_pre_hook(
            lambda module, args: shapes.extend(tuple(x.shape) for x in args[0])
        )
        model.eval()
        try:
            with torch.inference_mode():
                prediction = model(torch.zeros(1, 3, 640, 640))
        finally:
            handle.remove()
        assert shapes == [(1, 240, 80, 80), (1, 480, 40, 40), (1, 480, 20, 20)]
        prediction = prediction[0] if isinstance(prediction, tuple) else prediction
        assert prediction.shape == (1, export_limit, 6)
        assert torch.isfinite(prediction).all()
        restored = DetectionModel(cfg, verbose=False)
        result = restored.load_state_dict(model.state_dict(), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
    finally:
        torch.set_num_threads(previous_threads)
