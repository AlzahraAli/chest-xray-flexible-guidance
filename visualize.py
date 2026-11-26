from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from evaluate import upsample_and_normalize_map


METHODS = ["cls_only", "binary", "fixed_padding", "distance_weighted"]
CONDITIONS = ["unchanged", "translate", "shrink", "enlarge"]
METRICS = ["auroc", "f1", "mean_iou", "hit_rate"]


def summarize_results(
    records: list[dict],
    split: str = "test",
) -> list[dict]:
    """
    Aggregate per-seed metrics into mean and standard deviation per (method, condition).

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
        for cond in CONDITIONS:
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
) -> list[Path]:
    """Generate grouped bar charts across perturbation conditions for each metric."""
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

    x = np.arange(len(active_conditions))
    width = 0.8 / max(len(active_methods), 1)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for idx, method in enumerate(active_methods):
            means = []
            stds = []
            for cond in active_conditions:
                match = next(
                    (r for r in summary_rows if r["method"] == method and r["condition"] == cond),
                    None,
                )
                means.append(match[f"{metric}_mean"] if match else 0.0)
                stds.append(match[f"{metric}_std"] if match else 0.0)

            offset = (idx - (len(active_methods) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                label=method,
            )

        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Training Box Perturbation Condition")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_xticks(x)
        ax.set_xticklabels(active_conditions)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        save_file = out_path / f"compare_{metric}.png"
        fig.savefig(save_file, dpi=200)
        plt.close(fig)
        saved_paths.append(save_file)

    return saved_paths


def plot_heatmap_comparison(
    image: torch.Tensor | np.ndarray,
    activation_map: torch.Tensor,
    gt_boxes: torch.Tensor,
    save_path: str | Path,
    image_size: int = 320,
    title: str = "",
) -> Path:
    """
    Plot original image, upsampled normalized heatmap overlay, and ground-truth
    normalized [x_min, y_min, x_max, y_max] boxes side by side.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    # Handle [C, H, W] -> [H, W, C]
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)

    # Normalize image display values to [0, 1] for clean viewing
    img_disp = img_np.astype(np.float32)
    if img_disp.max() > img_disp.min():
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())

    hmap = upsample_and_normalize_map(activation_map, image_size=image_size)
    boxes_np = gt_boxes.detach().cpu().numpy() if isinstance(gt_boxes, torch.Tensor) else np.asarray(gt_boxes)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Left panel: Original X-ray with ground-truth boxes
    axes[0].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)
    for box in boxes_np:
        if len(box) == 4:
            x1, y1, x2, y2 = box * image_size
            rect = patches.Rectangle(
                (x1, y1), max(x2 - x1, 1.0), max(y2 - y1, 1.0),
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            axes[0].add_patch(rect)
    axes[0].set_title("Input + GT Boxes")
    axes[0].axis("off")

    # Right panel: Upsampled normalized heatmap overlay + GT boxes
    axes[1].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)
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
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    # Quick synthetic end-to-end visualization test
    rng = np.random.default_rng(42)
    sample_records = []
    for method in METHODS:
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

    fake_img = torch.rand((3, 320, 320))
    fake_amap = torch.zeros((10, 10))
    fake_amap[2:6, 2:6] = 1.0
    fake_boxes = torch.tensor([[0.2, 0.2, 0.6, 0.6]])
    overlay_file = plot_heatmap_comparison(
        fake_img,
        fake_amap,
        fake_boxes,
        save_path="outputs_test/sample_heatmap.png",
        title="Synthetic Test Overlay",
    )

    print("Saved CSV:", csv_file)
    print("Saved MD :", md_file)
    print("Saved charts:", [str(p) for p in chart_files])
    print("Saved overlay:", overlay_file)