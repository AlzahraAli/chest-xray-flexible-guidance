from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from evaluate import upsample_and_normalize_map


METHODS = ["cls_only", "binary", "fixed_padding", "distance_weighted"]
CONDITIONS = ["unchanged", "translate", "shrink", "enlarge"]
METRICS = ["auroc", "f1", "mean_iou", "hit_rate"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_cls_only_baseline_runs(
    grouped: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    """
    Return the classification-only baseline runs from 'unchanged'
    (or fallback to whichever single condition they were logged under).
    """
    if grouped.get(("cls_only", "unchanged")):
        return grouped[("cls_only", "unchanged")]
    for (method, _cond), runs in grouped.items():
        if method == "cls_only" and runs:
            return runs
    return []


def summarize_results(
    records: list[dict],
    split: str = "test",
) -> list[dict]:
    """
    Aggregate per-seed metrics into mean and standard deviation per (method, condition).
    Automatically reuses `cls_only` baseline runs across all active annotation conditions
    because classification-only models do not depend on training bounding boxes.

    Each item in `records` should look like:
        {
            "method": "distance_weighted",
            "condition": "shrink",
            "seed": 0,
            "val": {...},
            "test": {"auroc": ..., "f1": ..., "mean_iou": ..., "hit_rate": ...},
        }
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        key = (rec["method"], rec["condition"])
        grouped.setdefault(key, []).append(rec[split])

    active_conditions = [
        c for c in CONDITIONS if any(rec["condition"] == c for rec in records)
    ]
    cls_only_baseline = _get_cls_only_baseline_runs(grouped)

    if not active_conditions and cls_only_baseline:
        active_conditions = ["unchanged"]

    # Reuse cls_only baseline runs across all active annotation conditions
    if cls_only_baseline:
        for cond in active_conditions:
            if not grouped.get(("cls_only", cond)):
                grouped[("cls_only", cond)] = cls_only_baseline

    # Baseline lookup per method on "unchanged" condition for robustness delta
    unchanged_means: dict[str, dict[str, float]] = {}
    for method in METHODS:
        u_runs = grouped.get((method, "unchanged"), [])
        unchanged_means[method] = {}
        for m in METRICS:
            vals = [r[m] for r in u_runs if not np.isnan(r[m])]
            unchanged_means[method][m] = float(np.mean(vals)) if vals else float("nan")

    summary_rows: list[dict] = []
    for method in METHODS:
        for cond in active_conditions:
            runs = grouped.get((method, cond), [])
            if not runs:
                continue
            row: dict = {
                "method": method,
                "condition": cond,
                "n_seeds": len(runs),
            }
            for m in METRICS:
                vals = np.array([r[m] for r in runs], dtype=np.float64)
                valid = vals[~np.isnan(vals)]
                mean_val = float(np.mean(valid)) if len(valid) > 0 else float("nan")
                std_val = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                row[f"{m}_mean"] = mean_val
                row[f"{m}_std"] = std_val
                row[f"{m}_formatted"] = (
                    f"{mean_val:.4f} ± {std_val:.4f}" if not np.isnan(mean_val) else "N/A"
                )
                base_val = unchanged_means[method][m]
                if cond == "unchanged" or np.isnan(mean_val) or np.isnan(base_val):
                    row[f"{m}_delta"] = 0.0 if cond == "unchanged" else float("nan")
                else:
                    row[f"{m}_delta"] = mean_val - base_val
            summary_rows.append(row)

    return summary_rows


def save_summary_tables(
    summary_rows: list[dict],
    out_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    """Save summary metrics to CSV and a report-ready Markdown table."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "results_summary.csv"
    md_path = out_path / "results_summary.md"

    headers = ["Method", "Condition", "Seeds", "AUROC", "F1", "Mean IoU", "Hit Rate"]
    csv_lines = [",".join(headers)]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    for r in summary_rows:
        cells = [
            r["method"],
            r["condition"],
            str(r["n_seeds"]),
            r["auroc_formatted"],
            r["f1_formatted"],
            r["mean_iou_formatted"],
            r["hit_rate_formatted"],
        ]
        csv_lines.append(",".join(cells))
        md_lines.append("| " + " | ".join(cells) + " |")

    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def plot_metric_comparisons(
    summary_rows: list[dict],
    out_dir: str | Path = "outputs",
    dpi: int = 300,
) -> list[Path]:
    """
    Generate 300 DPI grouped bar charts across perturbation conditions for each metric.
    Reuses the `cls_only` baseline across all conditions and leaves genuinely missing
    entries as NaN instead of plotting misleading zeros.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    metric_titles = {
        "auroc": "Test AUROC Across Box Perturbations",
        "f1": "Test F1-Score Across Box Perturbations",
        "mean_iou": "Test Mean IoU Across Box Perturbations",
        "hit_rate": "Pointing-Game Hit Rate Across Box Perturbations",
    }

    active_conditions = [c for c in CONDITIONS if any(r["condition"] == c for r in summary_rows)]
    active_methods = [m for m in METHODS if any(r["method"] == m for r in summary_rows)]
    if not active_conditions or not active_methods:
        return saved_paths

    # Fallback row for cls_only in case summary_rows only contains its single baseline row
    cls_only_fallback = next(
        (r for r in summary_rows if r["method"] == "cls_only" and r["condition"] == "unchanged"),
        next((r for r in summary_rows if r["method"] == "cls_only"), None),
    )

    x = np.arange(len(active_conditions))
    width = 0.8 / max(len(active_methods), 1)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for idx, method in enumerate(active_methods):
            means = []
            stds = []
            for cond in active_conditions:
                match = next(
                    (r for r in summary_rows if r["method"] == method and r["condition"] == cond),
                    None,
                )
                if match is None and method == "cls_only":
                    match = cls_only_fallback

                means.append(match[f"{metric}_mean"] if match is not None else np.nan)
                stds.append(match[f"{metric}_std"] if match is not None else np.nan)

            offset = (idx - (len(active_methods) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                label=method,
            )

        ax.set_title(metric_titles[metric], pad=28)
        ax.set_xlabel("Training Box Perturbation Condition")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_xticks(x)
        ax.set_xticklabels(active_conditions)
        ax.set_ylim(0.0, 1.05)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(active_methods),
            frameon=False,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        save_file = out_path / f"compare_{metric}.png"
        fig.savefig(save_file, dpi=dpi)
        plt.close(fig)
        saved_paths.append(save_file)

    return saved_paths


def prepare_display_grayscale(
    image: torch.Tensor | np.ndarray,
    imagenet_normalized: bool = True,
) -> np.ndarray:
    """
    Convert an input image tensor/array ([3, H, W], [H, W, 3], [1, H, W], or [H, W])
    into a clean 2D grayscale image in [0, 1], reversing ImageNet normalization when enabled.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy().astype(np.float32)
    else:
        img_np = np.asarray(image, dtype=np.float32)

    # Convert channel-first [C, H, W] -> channel-last [H, W, C]
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
        img_np = np.transpose(img_np, (1, 2, 0))

    if img_np.ndim == 3 and img_np.shape[-1] == 3:
        if imagenet_normalized:
            img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
        elif img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0.0, 1.0)
        # Replicated grayscale channels recover identical values after unnormalization
        img_gray = img_np.mean(axis=-1)
    else:
        if img_np.ndim == 3 and img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_gray = np.clip(img_np, 0.0, 1.0)

    return img_gray


def plot_heatmap_comparison(
    image: torch.Tensor | np.ndarray,
    activation_map: torch.Tensor,
    gt_boxes: torch.Tensor,
    save_path: str | Path,
    image_size: int = 320,
    title: str = "",
    imagenet_normalized: bool = True,
    dpi: int = 300,
) -> Path:
    """
    Plot the unnormalized grayscale X-ray with ground-truth boxes alongside its
    upsampled normalized heatmap overlay at 300 DPI.
    """
    img_gray = prepare_display_grayscale(image, imagenet_normalized=imagenet_normalized)
    hmap = upsample_and_normalize_map(activation_map, image_size=image_size)
    boxes_np = gt_boxes.detach().cpu().numpy() if isinstance(gt_boxes, torch.Tensor) else np.asarray(gt_boxes)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Left panel: Original grayscale X-ray with ground-truth boxes
    axes[0].imshow(img_gray, cmap="gray", vmin=0.0, vmax=1.0)
    for box in boxes_np:
        if len(box) == 4:
            x1, y1, x2, y2 = box * image_size
            rect = patches.Rectangle(
                (x1, y1), max(x2 - x1, 1.0), max(y2 - y1, 1.0),
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            axes[0].add_patch(rect)
    axes[0].set_title("Input X-Ray + GT Boxes")
    axes[0].axis("off")

    # Right panel: Upsampled normalized heatmap overlay + GT boxes
    axes[1].imshow(img_gray, cmap="gray", vmin=0.0, vmax=1.0)
    im = axes[1].imshow(hmap, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    for box in boxes_np:
        if len(box) == 4:
            x1, y1, x2, y2 = box * image_size
            rect = patches.Rectangle(
                (x1, y1), max(x2 - x1, 1.0), max(y2 - y1, 1.0),
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            axes[1].add_patch(rect)
    axes[1].set_title("Normalized Activation Map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    dest = Path(save_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    # Realistic 39-run synthetic test:
    # cls_only is logged ONLY under "unchanged" (3 seeds), guided methods across all 4 conditions (36 runs)
    rng = np.random.default_rng(42)
    sample_records = []

    for seed in [0, 1, 2]:
        sample_records.append({
            "method": "cls_only",
            "condition": "unchanged",
            "seed": seed,
            "test": {
                "auroc": float(np.clip(0.84 + rng.normal(0, 0.01), 0.0, 1.0)),
                "f1": float(np.clip(0.70 + rng.normal(0, 0.02), 0.0, 1.0)),
                "mean_iou": float(np.clip(0.32 + rng.normal(0, 0.02), 0.0, 1.0)),
                "hit_rate": float(np.clip(0.66 + rng.normal(0, 0.02), 0.0, 1.0)),
            },
        })

    for method in ["binary", "fixed_padding", "distance_weighted"]:
        for cond in CONDITIONS:
            for seed in [0, 1, 2]:
                sample_records.append({
                    "method": method,
                    "condition": cond,
                    "seed": seed,
                    "test": {
                        "auroc": float(np.clip(0.85 + rng.normal(0, 0.02), 0.0, 1.0)),
                        "f1": float(np.clip(0.72 + rng.normal(0, 0.03), 0.0, 1.0)),
                        "mean_iou": float(np.clip(0.42 + rng.normal(0, 0.04), 0.0, 1.0)),
                        "hit_rate": float(np.clip(0.78 + rng.normal(0, 0.04), 0.0, 1.0)),
                    },
                })

    rows = summarize_results(sample_records, split="test")
    csv_file, md_file = save_summary_tables(rows, out_dir="outputs_test")
    chart_files = plot_metric_comparisons(rows, out_dir="outputs_test")

    # Build a synthetic ImageNet-normalized grayscale [3, 320, 320] tensor to test unnormalization
    gray_grad = np.linspace(0.0, 1.0, 320 * 320, dtype=np.float32).reshape(320, 320)
    rgb_3ch = np.stack([gray_grad, gray_grad, gray_grad], axis=0)
    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fake_norm_xray = (torch.from_numpy(rgb_3ch) - mean_t) / std_t

    fake_amap = torch.zeros((10, 10))
    fake_amap[2:6, 2:6] = 1.0
    fake_boxes = torch.tensor([[0.2, 0.2, 0.6, 0.6]])
    overlay_file = plot_heatmap_comparison(
        fake_norm_xray,
        fake_amap,
        fake_boxes,
        save_path="outputs_test/sample_heatmap.png",
        title="Synthetic Test Overlay (Unnormalized Grayscale)",
    )

    print(f"Total input records: {len(sample_records)} | Summary rows: {len(rows)}")
    print("Saved CSV:", csv_file)
    print("Saved MD :", md_file)
    print("Saved charts (300 DPI):", [str(p) for p in chart_files])
    print("Saved overlay (300 DPI):", overlay_file)