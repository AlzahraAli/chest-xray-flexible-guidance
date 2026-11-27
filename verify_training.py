"""Read-only audit of the CS421 training outputs. Does not access test images."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch

from model import ChestXRayClassifier

DATA_DIR = Path.home() / 'cs421-data'
RUNS = DATA_DIR / 'runs'
SEEDS = (42, 43, 44)
METHODS = ('binary', 'fixed_padding', 'distance_weighted')
CONDITIONS = ('unchanged', 'translate', 'shrink', 'enlarge')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_run(name, seed, model, positive_ids, method=None,
              condition=None, lam=None):
    folder = RUNS / name
    history = pd.read_csv(folder / 'history.csv')
    require(history['epoch'].tolist() == [1, 2, 3, 4, 5],
            f'{name}: incomplete or invalid epoch history')
    columns = ['train_loss', 'train_auroc', 'val_loss', 'val_auroc', 'seconds']
    if method is not None:
        columns += ['train_cls_loss', 'train_spatial_loss']
    for column in columns:
        require(all(math.isfinite(float(v)) for v in history[column]),
                f'{name}: nonfinite {column}')
    for column in ('train_auroc', 'val_auroc'):
        require(history[column].between(0, 1).all(), f'{name}: invalid {column}')
    require((history['seconds'] > 0).all(), f'{name}: invalid timings')
    best = history.loc[history['val_auroc'].idxmax()]
    expected = dict(seed=seed, batch_size=16, learning_rate=1e-4,
                    mixed_precision=True)
    checkpoint = torch.load(folder / 'best_model.pt', map_location='cpu',
                            weights_only=True)
    for key, value in expected.items():
        require(checkpoint[key] == value, f'{name}: checkpoint {key} mismatch')
    require(checkpoint['epoch'] == int(best['epoch']), f'{name}: wrong best epoch')
    require(math.isclose(float(checkpoint['val_auroc']), float(best['val_auroc']),
                         rel_tol=0, abs_tol=1e-12), f'{name}: checkpoint AUROC mismatch')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    for key, tensor in checkpoint['model_state_dict'].items():
        require(torch.isfinite(tensor).all().item(), f'{name}: nonfinite weights: {key}')
    box_hash = None
    if method is not None:
        guidance = dict(method=method, condition=condition, lam=lam, rho=0.05,
                        pad_frac=0.10)
        for key, value in dict(guidance, max_epochs=5).items():
            require(checkpoint[key] == value, f'{name}: checkpoint {key} mismatch')
        config = read_json(folder / 'config.json')
        for key, value in dict(expected, **guidance, epochs=5, workers=2,
                               patience=5, feasibility=False,
                               validation_loss='classification_only').items():
            require(config[key] == value, f'{name}: config {key} mismatch')
        box_path = folder / 'training_boxes.json'
        boxes = read_json(box_path)
        require(set(boxes) == positive_ids, f'{name}: positive image ID mismatch')
        for image_id, regions in boxes.items():
            require(len(regions) > 0, f'{name}: empty boxes for {image_id}')
            for region in regions:
                require(len(region) == 4, f'{name}: invalid box shape')
                x1, y1, x2, y2 = region
                require(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1,
                        f'{name}: invalid box coordinates for {image_id}')
        box_hash = digest(box_path)
        residual = (history['train_loss'] - history['train_cls_loss']
                    - lam * history['train_spatial_loss']).abs()
        require((residual < 1e-5).all(), f'{name}: inconsistent loss components')
    print(f'PASS {name} | best epoch {int(best["epoch"])} | AUROC {best["val_auroc"]:.8f}', flush=True)
    return box_hash, float(best['val_auroc'])


def main():
    torch.set_num_threads(1)
    train = pd.read_csv(DATA_DIR / 'splits' / 'train.csv')
    positive_ids = set(train.loc[train['target'] == 1, 'rsna_id'])
    require(len(train) == 21456 and len(positive_ids) == 4885,
            'Training split counts do not match the recorded counts')
    model = ChestXRayClassifier(pretrained=False)
    hashes = {}
    selected = {}
    for seed in SEEDS:
        audit_run(f'baseline_seed_{seed}', seed, model, positive_ids)
        for method in METHODS:
            for condition in CONDITIONS:
                name = f'{method}_{condition}_lam1_seed{seed}_ep5'
                h, auc = audit_run(name, seed, model, positive_ids, method, condition, 1.0)
                hashes[seed, method, condition] = h
                if seed == 42 and condition == 'unchanged':
                    selected[method] = auc
        for condition in CONDITIONS:
            require(len({hashes[seed, m, condition] for m in METHODS}) == 1,
                    f'Seed {seed}, {condition}: annotation mismatch')
        print(f'Seed {seed}: 13 primary runs and 4 annotation checks passed.', flush=True)
    for method in METHODS:
        name = f'{method}_unchanged_lam0.1_seed42_ep5'
        h, auc = audit_run(name, 42, model, positive_ids, method, 'unchanged', 0.1)
        require(h == hashes[42, method, 'unchanged'], f'{name}: tuning annotation mismatch')
        require(selected[method] > auc, f'{method}: lambda selection mismatch')
    for condition in ('unchanged', 'shrink', 'enlarge'):
        require(len({hashes[s, 'binary', condition] for s in SEEDS}) == 1,
                f'{condition}: deterministic annotations differ across seeds')
    require(len({hashes[s, 'binary', 'translate'] for s in SEEDS}) == 3,
            'Translated annotation files are not distinct across seeds')
    print('\nFINAL AUDIT PASSED')
    print('Primary runs: 39/39')
    print('Additional lambda runs: 3/3')
    print('Checkpoint and epoch checks: 42/42')
    print('Within-seed annotation checks: 12/12')
    print('Cross-seed annotation checks: Passed')
    print('Lambda selection checks: 3/3')
    print('No training or test-set evaluation performed.')


if __name__ == '__main__':
    main()
