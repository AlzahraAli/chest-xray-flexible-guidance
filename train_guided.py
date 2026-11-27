
import argparse
import hashlib
import json
import random
import time
from itertools import islice

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import RSNADataset, collate_fn, DATA_DIR
from model import ChestXRayClassifier
from losses import GuidedClassificationLoss
from perturbations import perturb_boxes


LEARNING_RATE = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class FixedTrainingBoxes:
    def __init__(self, condition, seed):
        self.condition = condition
        self.seed = seed
        self.cache = {}

    def get(self, image_id, boxes):
        if boxes.numel() == 0:
            return boxes
        if image_id not in self.cache:
            key = f"{self.seed}:{image_id}".encode("utf-8")
            image_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            rng = np.random.default_rng(image_seed)
            self.cache[image_id] = perturb_boxes(boxes, self.condition, rng)
        return self.cache[image_id]

    def save(self, path):
        records = {image_id: boxes.tolist()
                   for image_id, boxes in sorted(self.cache.items())}
        path.write_text(json.dumps(records, sort_keys=True), encoding="utf-8")


def run_epoch(model, loader, criterion, optimizer,
              scaler, training, limit=None, fixed_boxes=None):

    model.train(training)

    total_loss = 0.0
    total_cls = 0.0
    total_spatial = 0.0
    total_images = 0
    targets_all = []
    scores_all = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    batches = loader if limit is None else islice(loader, limit)

    with torch.set_grad_enabled(training):
        for step, (images, targets, boxes, image_ids) in enumerate(
            batches, start=1
        ):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if training:
                boxes = [fixed_boxes.get(image_id, box)
                         for image_id, box in zip(image_ids, boxes)]
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits, activation = model(images)
                if training:
                    parts = criterion(logits, activation, targets, boxes)
                    loss = parts["total"]
                    cls_loss = parts["cls"]
                    spatial_loss = parts["spatial"]
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, targets
                    )
                    cls_loss = loss
                    spatial_loss = loss.new_zeros(())

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss detected.")

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(images)
            total_cls += cls_loss.item() * len(images)
            total_spatial += spatial_loss.item() * len(images)
            total_images += len(images)

            targets_all.extend(targets.detach().cpu().tolist())
            scores_all.extend(logits.detach().float().cpu().tolist())

            if step % 10 == 0 and limit is not None:
                print(f"Batch {step}: loss={loss.item():.4f}", flush=True)

            elif step % 100 == 0 and training:
                print(
                    f"Batch {step}/{len(loader)}: "
                    f"loss={loss.item():.4f}",
                    flush=True
                )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mean_loss = total_loss / total_images

    auc = None

    if limit is None and len(set(targets_all)) == 2:
        auc = roc_auc_score(targets_all, scores_all)

    return (mean_loss, auc, elapsed, total_images,
            total_cls / total_images, total_spatial / total_images)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--feasibility", action="store_true")

    parser.add_argument("--method", required=True,
                        choices=["binary", "fixed_padding", "distance_weighted"])
    parser.add_argument("--condition", default="unchanged",
                        choices=["unchanged", "translate", "shrink", "enlarge"])
    parser.add_argument("--lam", required=True, type=float, choices=[0.1, 1.0])

    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers must be nonnegative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    set_seed(args.seed)

    train_data = RSNADataset("train")
    val_data = RSNADataset("val")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "persistent_workers": args.workers > 0
    }

    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        **loader_options
    )

    model = ChestXRayClassifier(pretrained=True).to(DEVICE)

    criterion = GuidedClassificationLoss(
        method=args.method, lam=args.lam, rho=0.05, pad_frac=0.10
    )
    fixed_boxes = FixedTrainingBoxes(args.condition, args.seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = torch.amp.GradScaler("cuda")

    print("Method:", args.method)
    print("Condition:", args.condition)
    print("Lambda:", args.lam)
    print("Seed:", args.seed)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))
    print("Batch size:", args.batch_size)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    if args.feasibility:
        print("\nFeasibility test started.")

        torch.cuda.reset_peak_memory_stats()

        train_loss, _, train_time, train_count, train_cls, train_spatial = run_epoch(
            model, train_loader, criterion,
            optimizer, scaler, True, limit=30, fixed_boxes=fixed_boxes
        )

        val_loss, _, val_time, val_count, _, _ = run_epoch(
            model, val_loader, criterion,
            optimizer, scaler, False, limit=10
        )

        estimated_epoch = (
            train_time / train_count * len(train_data)
            + val_time / val_count * len(val_data)
        )

        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        print("\nFeasibility results")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training classification loss: {train_cls:.4f}")
        print(f"Training spatial loss (unweighted): {train_spatial:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Training images processed: {train_count}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Validation images processed: {val_count}")
        print(f"Validation time: {val_time:.2f} seconds")
        print(f"Peak GPU allocation: {peak_memory:.2f} GiB")
        print(f"Estimated epoch time: {estimated_epoch / 60:.2f} minutes")
        print("Feasibility test completed.")

        return

    run_name = f"{args.method}_{args.condition}_lam{args.lam:g}_seed{args.seed}_ep{args.epochs}"
    output_dir = DATA_DIR / "runs" / run_name

    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), learning_rate=LEARNING_RATE, patience=PATIENCE,
                  mixed_precision=True, rho=0.05, pad_frac=0.10,
                  validation_loss="classification_only")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_auc = -float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        train_loss, train_auc, train_time, _, train_cls, train_spatial = run_epoch(
            model, train_loader, criterion,
            optimizer, scaler, True, fixed_boxes=fixed_boxes
        )

        val_loss, val_auc, val_time, _, _, _ = run_epoch(
            model, val_loader, criterion,
            optimizer, scaler, False
        )

        print(f"Training loss: {train_loss:.4f}")
        print(f"Training classification loss: {train_cls:.4f}")
        print(f"Training spatial loss (unweighted): {train_spatial:.4f}")
        print(f"Training AUROC: {train_auc:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation AUROC: {val_auc:.4f}")
        print(f"Epoch time: {(train_time + val_time) / 60:.2f} min")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_cls,
            "train_spatial_loss": train_spatial,
            "train_auroc": train_auc,
            "val_loss": val_loss,
            "val_auroc": val_auc,
            "seconds": train_time + val_time
        })

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False
        )

        if epoch == 1:
            fixed_boxes.save(output_dir / "training_boxes.json")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auroc": val_auc,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": LEARNING_RATE,
                "mixed_precision": True,
                "method": args.method,
                "condition": args.condition,
                "lam": args.lam,
                "rho": 0.05,
                "pad_frac": 0.10,
                "max_epochs": args.epochs
            }, output_dir / "best_model.pt")

            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    print(f"\nBest validation AUROC: {best_auc:.4f}")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
