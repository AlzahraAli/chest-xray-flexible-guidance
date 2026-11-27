"""Verify and aggregate the 39 completed evaluations; no model inference."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from run_evaluation import primary_runs, verify_split_hash, sha256
from visualize import summarize_results, save_summary_tables, plot_metric_comparisons

METRICS = ('auroc', 'f1', 'mean_iou', 'hit_rate')
COUNTS = {'val': 585, 'test': 542}
SEEDS = (42, 43, 44)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def collect(data_dir, code_root, split_hashes):
    records = []
    expected = list(primary_runs())
    require(len(expected) == 39, 'Unexpected primary-run count')
    source_hashes = {name: sha256(code_root / name) for name in
                     ('run_evaluation.py', 'evaluate.py', 'dataset.py', 'model.py')}
    for name, method, condition, seed in expected:
        folder = data_dir / 'evaluation' / name
        record = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        expected_meta = dict(run=name, method=method, condition=condition, seed=seed,
                             **{'lambda': 0.0 if method == 'cls_only' else 1.0})
        require(all(record.get(k) == v for k, v in expected_meta.items()), f'{name}: metadata mismatch')
        require(record['split_sha256'] == split_hashes, f'{name}: split hashes mismatch')
        require(record['source_sha256'] == source_hashes, f'{name}: evaluation source changed')
        require(sha256(folder / 'predictions.pt') == record['predictions_sha256'], f'{name}: prediction hash mismatch')
        require(sha256(data_dir / 'runs' / name / 'best_model.pt') == record['checkpoint_sha256'],
                f'{name}: checkpoint hash mismatch')
        history = pd.read_csv(data_dir / 'runs' / name / 'history.csv')
        best = history.loc[history['val_auroc'].idxmax()]
        require(len(history) == 5 and list(history['epoch']) == [1, 2, 3, 4, 5], f'{name}: incomplete history')
        require(record['best_epoch'] == int(best['epoch']), f'{name}: checkpoint epoch mismatch')
        require(abs(record['checkpoint_val_auroc'] - float(best['val_auroc'])) < 1e-10,
                f'{name}: checkpoint AUROC mismatch')
        require(abs(record['val']['auroc'] - record['checkpoint_val_auroc']) <= 0.0001,
                f'{name}: validation AUROC mismatch')
        for split, count in COUNTS.items():
            metrics = record[split]
            require(metrics['num_annotated_positives'] == count, f'{name}: {split} positive count mismatch')
            for metric in METRICS:
                value = metrics[metric]
                require(math.isfinite(value) and 0 <= value <= 1, f'{name}: invalid {split} {metric}')
        for key, grid in [('cls_threshold', np.linspace(.05, .95, 91)),
                          ('heatmap_threshold', np.arange(1, 10) / 10)]:
            value = record['val'][key]
            require(value == record['test'][key] and np.any(np.isclose(value, grid, atol=1e-10, rtol=0)),
                    f'{name}: invalid or inconsistent {key}')
        records.append(record)
    return records


def aggregate(records):
    rows = summarize_results(records, split='test')
    require(len(rows) == 16, 'Expected 16 summary rows (including reused baseline)')
    require(all(r['n_seeds'] == 3 for r in rows), 'Incomplete seed group')
    lookup = {(r['method'], r['condition'], r['seed']): r for r in records}
    require(len(lookup) == 39, 'Duplicate or missing primary run')
    paired = []
    for row in rows:
        m, c = row['method'], row['condition']
        actual_c = 'unchanged' if m == 'cls_only' else c
        for metric in METRICS:
            values = np.array([lookup[m, actual_c, s]['test'][metric] for s in SEEDS])
            baseline = np.array([lookup[m, 'unchanged', s]['test'][metric] for s in SEEDS])
            delta = values - baseline
            require(np.isclose(row[f'{metric}_mean'], values.mean(), atol=1e-12, rtol=0), 'Mean mismatch')
            require(np.isclose(row[f'{metric}_std'], values.std(ddof=1), atol=1e-12, rtol=0), 'SD mismatch')
            require(np.isclose(row[f'{metric}_delta'], delta.mean(), atol=1e-12, rtol=0), 'Delta mismatch')
            for s, d in zip(SEEDS, delta):
                paired.append(dict(method=m, condition=c, seed=s, metric=metric, delta=float(d),
                                   baseline_reused=m == 'cls_only' and c != 'unchanged'))
        row['baseline_reused'] = m == 'cls_only' and c != 'unchanged'
    return rows, paired


def main():
    from dataset import DATA_DIR
    data_dir = Path(DATA_DIR)
    code_root = Path(__file__).resolve().parent
    split_hashes = {s: verify_split_hash(data_dir, s) for s in COUNTS}
    records = collect(data_dir, code_root, split_hashes)
    rows, paired = aggregate(records)
    out = data_dir / 'final_results'
    out.mkdir(exist_ok=False)
    per_run = []
    for r in records:
        base = {k: r[k] for k in ('run', 'method', 'condition', 'seed', 'lambda', 'best_epoch')}
        for split in COUNTS:
            per_run.append(dict(base, split=split, **r[split]))
    pd.DataFrame(per_run).to_csv(out / 'per_run_metrics.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'summary_numeric.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'paired_deltas.csv', index=False)
    save_summary_tables(rows, out)
    plots = plot_metric_comparisons(rows, out, dpi=300)
    require(len(plots) == 4 and all(p.is_file() for p in plots), 'Missing comparison figures')
    print((out / 'results_summary.md').read_text(encoding='utf-8'))
    print('\nROBUSTNESS DELTAS (test mean minus unchanged mean)')
    columns = ['method', 'condition'] + [f'{m}_delta' for m in METRICS]
    print(pd.DataFrame(rows)[columns].to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
    print('\nAGGREGATION PASSED: 39 primary runs; 16 comparison rows; 4 charts.')
    print('Saved results:', out)
    print('No model inference or training performed.')


if __name__ == '__main__':
    main()
