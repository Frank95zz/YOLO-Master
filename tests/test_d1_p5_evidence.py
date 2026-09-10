"""Offline consistency checks for the completed P5 screen and its archival evidence."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'experiments/d1'
ARCHIVE = DOCS / 'manifests/p5-screen-20260909'
VARIANTS = ('BASE', 'DW', 'BN64')
CODE_COMMIT = '7a0ad4dc32cd90222d37be0b3c24ed18a8e53c37'


def read(name):
    return json.loads((ARCHIVE / name).read_text(encoding='utf-8'))


def test_archive_checksums_and_original_copies():
    index = read('archive-index.json')
    actual = {p.name for p in ARCHIVE.iterdir() if p.is_file()}
    assert actual == set(index['files']) | {'archive-index.json'}
    assert index['training_code_commit'] == CODE_COMMIT
    for name, metadata in index['files'].items():
        assert Path(name).name == name
        raw = (ARCHIVE / name).read_bytes()
        assert len(raw) == metadata['bytes']
        assert hashlib.sha256(raw).hexdigest() == metadata['sha256']
    for name in ('suite-summary.json', 'suite-launch.json'):
        assert index['files'][name] == index['original_sources'][name]
    original = read('training-integrity.json')['source_files']
    for variant in VARIANTS:
        assert index['files'][f'{variant}-results.csv'] == original[f'runs/P5-{variant}/results.csv']


def test_common_execution_contract():
    summary = read('suite-summary.json')
    launch = read('suite-launch.json')
    matrix = read('matrix-summary.json')
    assert summary['status'] == 'completed'
    assert summary['identity'] == launch['identity'] == matrix['identity']
    assert summary['identity']['commit'] == CODE_COMMIT
    assert launch['variants'] == list(VARIANTS)
    assert (launch['seed'], launch['screen_epochs'], launch['schedule_epochs']) == (0, 50, 100)
    assert (launch['world_size'], launch['global_batch'], launch['workers_per_rank']) == (6, 384, 4)
    assert launch['p3_upsample_mode'] == 'separable_bilinear2x'
    assert launch['ema_implementation'] == 'foreach-v1'
    assert launch['resume_rank_buffers'] is True
    assert launch['resume_temperature_policy'] == 'epoch-boundary-v1'


@pytest.mark.parametrize('variant', VARIANTS)
def test_budget_metrics_and_rank_integrity(variant):
    row = read('suite-summary.json')['results'][variant]
    audit = read('training-integrity.json')['variants'][variant]
    assert (row['training']['status'], row['training']['epochs'], row['training']['optimizer_steps']) == (
        'completed', 50, 15450
    )
    assert len(audit['ranks']) == 6
    assert {r['rank'] for r in audit['ranks']} == set(range(6))
    for rank in audit['ranks']:
        assert (rank['epochs'], rank['final_optimizer_steps'], rank['max_amp_retries']) == (50, 15450, 0)
        assert rank['final_routing_finite']
        assert rank['mean_data_wait_seconds'] == pytest.approx(rank['sum_data_wait_seconds'] / 50)
    with (ARCHIVE / f'{variant}-results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r['epoch']) for r in rows] == list(range(1, 51))
    assert all(math.isfinite(float(value)) for r in rows for value in r.values())
    assert audit['csv_rows'] == 50 and audit['csv_numeric_finite']
    assert audit['resume_verified']['sha256'] == row['training']['resume_sha256']


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('checkpoint', ('last', 'standard-best'))
def test_final_evaluations_bind_to_real_artifacts(variant, checkpoint):
    summary = read('suite-summary.json')
    audit = read('training-integrity.json')
    report = summary['results'][variant]['final'][checkpoint]
    assert report['identity'] == report['execution_identity'] == summary['identity']
    assert report['seen'] == 5000 and report['status'] == 'passed'
    assert report['strict_reload'] and report['teacher_parameters'] == 0
    assert report['checkpoint_sha256'] == audit['variants'][variant]['checkpoints_verified'][checkpoint]['sha256']
    source = audit['source_files'][f'reports/P5-{variant}/final/{checkpoint}/predictions.json']
    assert report['prediction_sha256'] == source['sha256']
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in report['official'].values())
    expected_epoch = 50 if checkpoint == 'last' else {'BASE': 30, 'DW': 45, 'BN64': 35}[variant]
    assert report['checkpoint_epoch'] == expected_epoch


def test_cost_formula_and_unchanged_screening_gates():
    from scripts.d1.run_p5_suite import comparison

    summary = read('suite-summary.json')
    selection = read('matrix-summary.json')['contract']['selection']
    assert comparison(summary['results'], selection) == summary['comparison']
    assert summary['comparison']['BASE']['retained']
    for variant, row in summary['results'].items():
        final_seconds = sum(row['final_job_seconds'].values())
        assert row['active_job_GPUh'] == pytest.approx((6 * row['train_seconds'] + final_seconds) / 3600)
        assert row['six_gpu_reserved_hours'] == pytest.approx(6 * (row['train_seconds'] + final_seconds) / 3600)
        if variant != 'BASE':
            assert summary['comparison'][variant]['retained'] is False
            drops = summary['comparison'][variant]['drop_points_vs_BASE']
            assert drops['AP_all'] <= 0.5 and drops['AP_75'] <= 1.0 and drops['AP_large'] <= 1.0
            assert row['active_job_GPUh'] > summary['results']['BASE']['active_job_GPUh']


def test_overlap_is_a_task_window_not_subtracted_cost():
    evidence = read('resource-overlap.json')
    jobs = {j['label']: j for j in evidence['jobs']}
    assert len(jobs) == 9 and all(j['returncode'] == 0 for j in jobs.values())
    bn64 = jobs['P5-BN64']
    windows = []
    for row in sorted(evidence['logs'], key=lambda r: r['start_unix']):
        start, end = row['start_unix'], row['end_unix']
        assert bn64['started_unix'] <= start < end <= bn64['ended_unix']
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    assert windows == evidence['merged_task_windows']
    duration = sum(end - start for start, end in windows)
    assert duration == pytest.approx(evidence['task_window_union_seconds'])
    assert duration == pytest.approx(771.5661075115204)
    assert evidence['affected_variant'] == 'BN64' and evidence['limitations']
    for left, right in zip(VARIANTS, VARIANTS[1:]):
        assert jobs[f'P5-{left}-final-standard-best']['ended_unix'] <= jobs[f'P5-{right}']['started_unix']


def test_report_tables_match_unrounded_evidence():
    text = (DOCS / 'P5_FAST_RUN_20260909.md').read_text(encoding='utf-8')
    summary = read('suite-summary.json')
    for variant in VARIANTS:
        row = summary['results'][variant]
        values = [f'{row["downstream_parameters"]:,}']
        values += [f'{100 * row["final"]["last"]["official"][key]:.3f}' for key in (
            'AP_all', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large'
        )]
        assert '| ' + variant + ' | ' + ' | '.join(values) + ' |' in text


def test_report_links_and_archive_portability():
    text = (DOCS / 'P5_FAST_RUN_20260909.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if not target.startswith(('https://', 'http://', '#')):
            assert (DOCS / target.split('#')[0]).exists(), target
    for file in ARCHIVE.iterdir():
        content = file.read_text(encoding='utf-8')
        assert '/root/' not in content and '/data/yingxi/' not in content and '/localssd/' not in content
        assert 'PRIVATE KEY' not in content
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}', content)


def test_nonoverlap_selection_uses_time_windows_not_speed():
    timing = read('nonoverlap-timing.json')
    overlap = ARCHIVE / 'resource-overlap.json'
    assert timing['overlap_source_sha256'] == hashlib.sha256(overlap.read_bytes()).hexdigest()
    assert timing['source_code_commit'] == CODE_COMMIT
    rows = timing['variants']['BN64']['rows']
    excluded = []
    guarded = []
    for row in rows:
        hits = [i for i, (start, end) in enumerate(timing['fins_windows'])
                if row['start_unix'] < end and row['train_end_mtime_unix'] > start]
        assert row['overlap'] == hits
        if hits:
            excluded.append(row['epoch'])
        if any(row['start_unix'] < end + 1 and row['train_end_mtime_unix'] > start - 1
               for start, end in timing['fins_windows']):
            guarded.append(row['epoch'])
    assert excluded == timing['excluded_epochs'] == [23, 24, 25, 27, 28, 29, 30, 31]
    assert guarded == timing['method']['excluded_with_guard'] == excluded
    included = [epoch for epoch in range(1, 51) if epoch not in excluded]
    assert included == timing['included_epochs'] and len(included) == 42
    assert [epoch for epoch in included if epoch > 3] == timing['post_warmup_included_epochs']


@pytest.mark.parametrize('variant', VARIANTS)
def test_nonoverlap_statistics_recompute_from_all_recorded_epochs(variant):
    import statistics

    timing = read('nonoverlap-timing.json')
    source = timing['variants'][variant]
    assert source['source_file_count'] == 350
    assert re.fullmatch('[0-9a-f]{64}', source['source_concat_sha256'])
    rows = source['rows']
    assert [r['epoch'] for r in rows] == list(range(1, 51))
    assert all(r['start_unix'] < r['train_end_mtime_unix'] <= r['fit_end_unix'] for r in rows)
    assert all(r['fit_wall_s'] == pytest.approx(r['fit_end_unix'] - r['start_unix']) for r in rows)
    selections = {
        'all_epochs': list(range(1, 51)),
        'matched_nonoverlap': timing['included_epochs'],
        'matched_nonoverlap_post_warmup': timing['post_warmup_included_epochs'],
        'excluded_epochs': timing['excluded_epochs'],
    }
    functions = {'mean': statistics.mean, 'median': statistics.median,
                 'minimum': min, 'maximum': max, 'sample_std': statistics.stdev}
    for label, epoch_ids in selections.items():
        selected = [r for r in rows if r['epoch'] in epoch_ids]
        summary = timing['summaries'][variant][label]
        assert summary['count'] == len(selected)
        for metric in ('step_s', 'wait_s', 'max_rank_loop_s', 'fit_wall_s'):
            values = [r[metric] for r in selected]
            assert all(math.isfinite(v) and v >= 0 for v in values)
            for name, function in functions.items():
                assert summary[metric][name] == pytest.approx(function(values))
    rank_records = read('training-integrity.json')['variants'][variant]['ranks']
    assert statistics.mean(r['mean_step_seconds'] for r in rank_records) == pytest.approx(
        timing['summaries'][variant]['all_epochs']['step_s']['mean']
    )


def test_nonoverlap_report_table_is_matched_and_does_not_replace_original_cost():
    timing = read('nonoverlap-timing.json')
    text = (DOCS / 'P5_FAST_RUN_20260909.md').read_text(encoding='utf-8')
    for variant in VARIANTS:
        row = timing['summaries'][variant]['matched_nonoverlap']
        values = [row['max_rank_loop_s']['mean'], row['max_rank_loop_s']['median'],
                  row['wait_s']['mean'], row['fit_wall_s']['mean']]
        line = '| ' + variant + ' | ' + ' | '.join(f'{value:.3f}' for value in values) + ' |'
        assert line in text
    original = read('suite-summary.json')['results']['BN64']
    assert original['active_job_GPUh'] == pytest.approx(10.424189264914022)
    assert read('suite-summary.json')['comparison']['BN64']['retained'] is False
