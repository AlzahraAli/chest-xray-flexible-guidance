"""Evaluate audited Phase II checkpoints without retraining or test-set tuning."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluate import (
    evaluate_split,
    select_classification_threshold,
    select_heatmap_threshold,
)

SEEDS = (42, 43, 44)
METHODS = ("binary", "fixed_padding", "distance_weighted")
CONDITIONS = ("unchanged", "translate", "shrink", "enlarge")
COUNTS = {"val": (2628, 585), "test": (2600, 542)}
METRICS = ("auroc", "f1", "mean_iou", "hit_rate")
DEVICE = torch.device("cuda")
# Same GPU inference precision and batch size as checkpoint selection.
AUC_TOLERANCE = 0.0001


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def primary_runs():
    for seed in SEEDS:
        yield (f"baseline_seed_{seed}", "cls_only", "unchanged", seed)
        for method in METHODS:
            for condition in CONDITIONS:
                yield (f"{method}_{condition}_lam1_seed{seed}_ep5", method, condition, seed)


def verify_split_hash(data_dir, split):
    evidence = Path(__file__).resolve().parent / "evidence" / "split_hashes.csv"
    saved = pd.read_csv(evidence, encoding="utf-8-sig")
    filename = f"{split}.csv"
    matches = saved.loc[saved["File"] == filename, "Hash"]
    require(len(matches) == 1, f"Missing or duplicate split hash: {filename}")
    current = sha256(data_dir / "splits" / filename)
    require(current.lower() == str(matches.iloc[0]).lower(), f"Split hash mismatch: {filename}")
    return current


def build_loader(data_dir, split, workers):
    from dataset import RSNADataset, collate_fn

    dataset = RSNADataset(split, data_dir=data_dir)
    records = dataset.records
    expected_total, expected_pos = COUNTS[split]
    require(len(dataset) == expected_total, f"{split}: unexpected image count")
    require(records["rsna_id"].is_unique, f"{split}: duplicate image IDs")
    require(records["target"].isin([0, 1]).all(), f"{split}: invalid labels")
    require(int(records["target"].sum()) == expected_pos, f"{split}: unexpected positive count")
    return DataLoader(dataset, batch_size=16, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, collate_fn=collate_fn)


def collect_predictions(model, loader, split):
    model.eval()
    labels_all, logits_all, maps_all, boxes_all, ids_all = [], [], [], [], []
    with torch.inference_mode():
        for images, labels, boxes, ids in loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, maps = model(images)
            require(tuple(logits.shape) == (len(ids),), "Unexpected logits shape")
            require(tuple(maps.shape) == (len(ids), 10, 10), "Unexpected activation shape")
            logits_all.append(logits.float().cpu())
            maps_all.append(maps.float().cpu())
            labels_all.append(labels.float().cpu())
            boxes_all.extend([box.detach().float().cpu() for box in boxes])
            ids_all.extend(list(ids))
    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    maps = torch.cat(maps_all)
    count, positives = COUNTS[split]
    require(len(ids_all) == count and len(set(ids_all)) == count, f"{split}: missing/duplicate predictions")
    require(ids_all == loader.dataset.records["rsna_id"].tolist(), f"{split}: image order mismatch")
    require(torch.equal(labels, torch.tensor(loader.dataset.records["target"].tolist(), dtype=torch.float32)),
            f"{split}: label mismatch")
    require(int(labels.sum().item()) == positives, f"{split}: positive count mismatch")
    require(torch.isfinite(logits).all().item(), f"{split}: nonfinite logits")
    require(torch.isfinite(maps).all().item() and (maps >= 0).all().item(), f"{split}: invalid activation maps")
    require(torch.allclose(maps.sum(dim=(1, 2)), torch.ones(count), atol=1e-5, rtol=0), f"{split}: activation sum mismatch")
    require(len(boxes_all) == count, f"{split}: missing boxes")
    for label, box in zip(labels.tolist(), boxes_all):
        require(box.ndim == 2 and box.shape[1] == 4, f"{split}: invalid box shape")
        require((box.numel() > 0) == (label == 1), f"{split}: label/box mismatch")
        if box.numel():
            require(torch.isfinite(box).all().item() and ((box >= 0) & (box <= 1)).all().item()
                    and (box[:, 2:] > box[:, :2]).all().item(), f"{split}: invalid box coordinates")
    return {"labels": labels, "logits": logits, "probs": logits.double().sigmoid(),
            "activation_maps": maps, "boxes": boxes_all, "image_ids": ids_all}


def load_model(model, path, method, condition, seed):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint["seed"] == seed and checkpoint["batch_size"] == 16,
            f"{path}: checkpoint metadata mismatch")
    require(checkpoint["mixed_precision"] is True, f"{path}: precision mismatch")
    if method != "cls_only":
        require(checkpoint["method"] == method and checkpoint["condition"] == condition
                and checkpoint["lam"] == 1.0, f"{path}: guidance metadata mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return int(checkpoint["epoch"]), float(checkpoint["val_auroc"])


def choose_thresholds(preds):
    # Double-precision probabilities ensure identical threshold comparisons in
    # validation selection and evaluation (including values on grid boundaries).
    cls, f1 = select_classification_threshold(preds["labels"], preds["probs"])
    heatmap, iou = select_heatmap_threshold(preds["labels"], preds["activation_maps"], preds["boxes"])
    return cls, heatmap, f1, iou


def metrics_for(preds, split, cls, heatmap):
    metrics = evaluate_split(preds["labels"], preds["probs"], preds["activation_maps"],
                             preds["boxes"], cls, heatmap)
    # AUROC is computed from raw logits to match checkpoint selection exactly;
    # sigmoid is monotonic but can create artificial ties in finite precision.
    metrics["auroc"] = float(roc_auc_score(preds["labels"].numpy(), preds["logits"].numpy()))
    require(metrics["num_annotated_positives"] == COUNTS[split][1], f"{split}: localization count mismatch")
    for key in METRICS:
        require(math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1, f"{split}: invalid metric {key}")
    return metrics


def validate_model(model, loader, saved_auc):
    preds = collect_predictions(model, loader, "val")
    cls, heatmap, f1, iou = choose_thresholds(preds)
    metrics = metrics_for(preds, "val", cls, heatmap)
    require(abs(metrics["f1"] - f1) < 1e-12, "Validation F1 threshold selection mismatch")
    require(abs(metrics["mean_iou"] - iou) < 1e-12, "Validation IoU threshold selection mismatch")
    difference = abs(metrics["auroc"] - saved_auc)
    print(f"Saved validation AUROC: {saved_auc:.8f}", flush=True)
    print(f"Recomputed validation AUROC: {metrics['auroc']:.8f} | difference: {difference:.8f}", flush=True)
    require(difference <= AUC_TOLERANCE, "Validation AUROC reproduction failed; investigate before test evaluation")
    print(f"Validation F1: {metrics['f1']:.4f} | classification threshold: {cls:.2f}", flush=True)
    print(f"Validation IoU: {metrics['mean_iou']:.4f} | heatmap threshold: {heatmap:.1f}", flush=True)
    print(f"Validation pointing-game hit rate: {metrics['hit_rate']:.4f}", flush=True)
    print(f"Validation images: {len(preds['image_ids'])} | annotated positives: {metrics['num_annotated_positives']}", flush=True)
    return preds, metrics, cls, heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validation-only smoke test of baseline and proposed guided model")
    mode.add_argument("--all", action="store_true", help="Evaluate all 39 primary runs after validation checks")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    require(args.workers >= 0, "Workers must be nonnegative")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    from dataset import DATA_DIR
    from model import ChestXRayClassifier

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    data_dir = Path(DATA_DIR)
    runs = list(primary_runs())
    require(len(runs) == 39, "Unexpected run count")
    if args.check:
        runs = [run for run in runs if run[0] in (
            "baseline_seed_42", "distance_weighted_translate_lam1_seed42_ep5")]
    val_hash = verify_split_hash(data_dir, "val")
    val_loader = build_loader(data_dir, "val", args.workers)
    model = ChestXRayClassifier(pretrained=False).to(DEVICE)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    for name, _, _, _ in runs:
        require((data_dir / "runs" / name / "best_model.pt").is_file(), f"Missing checkpoint: {name}")

    if args.all:
        # No test loader is constructed at all in --check mode.
        test_hash = verify_split_hash(data_dir, "test")
        test_loader = build_loader(data_dir, "test", args.workers)
        output = data_dir / "evaluation"
        output.mkdir(exist_ok=True)
        code_root = Path(__file__).resolve().parent
        code_hashes = {name: sha256(code_root / name) for name in (
            "run_evaluation.py", "evaluate.py", "dataset.py", "model.py")}

    for index, (name, method, condition, seed) in enumerate(runs, 1):
        print(f"\n[{index}/{len(runs)}] {name}", flush=True)
        checkpoint_path = data_dir / "runs" / name / "best_model.pt"
        if args.all:
            dest = output / name
            meta = {"run": name, "method": method, "condition": condition, "seed": seed,
                    "lambda": 0.0 if method == "cls_only" else 1.0,
                    "checkpoint_sha256": sha256(checkpoint_path), "split_sha256": {"val": val_hash, "test": test_hash},
                    "source_sha256": code_hashes}
            result_path = dest / "metrics.json"
            prediction_path = dest / "predictions.pt"
            if result_path.exists():
                old = json.loads(result_path.read_text(encoding="utf-8"))
                require(all(old.get(key) == value for key, value in meta.items()), f"{name}: existing evaluation metadata mismatch")
                require(prediction_path.is_file() and sha256(prediction_path) == old["predictions_sha256"],
                        f"{name}: existing prediction file missing or changed")
                print("Verified existing evaluation; skipping.", flush=True)
                continue
            require(not prediction_path.exists(), f"{name}: incomplete prior output; inspect before retry")
        epoch, saved_auc = load_model(model, checkpoint_path, method, condition, seed)
        val_preds, val_metrics, cls, heatmap = validate_model(model, val_loader, saved_auc)
        if args.check:
            continue
        # Both thresholds are fixed before any test-image inference for this run.
        test_preds = collect_predictions(model, test_loader, "test")
        test_metrics = metrics_for(test_preds, "test", cls, heatmap)
        dest.mkdir(exist_ok=True)
        torch.save({"val": val_preds, "test": test_preds}, prediction_path)
        result = dict(meta, best_epoch=epoch, checkpoint_val_auroc=saved_auc,
                      val=val_metrics, test=test_metrics, predictions_sha256=sha256(prediction_path))
        temp_path = dest / "metrics.json.tmp"
        temp_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        temp_path.replace(result_path)
        print("Test metrics:", " | ".join(f"{key}={test_metrics[key]:.4f}" for key in METRICS), flush=True)
        print(f"Test images: {len(test_preds['image_ids'])} | annotated positives: {test_metrics['num_annotated_positives']}", flush=True)

    if args.check:
        print("\nVALIDATION CHECK PASSED (2/2). No test images evaluated.")
    else:
        print("\nEVALUATION COMPLETE (39/39).")
        print("Saved results:", output)


if __name__ == "__main__":
    main()
