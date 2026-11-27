import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


HEATMAP_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def upsample_and_normalize_map(
    activation_map: torch.Tensor,
    image_size: int = 320,
) -> np.ndarray:
    """
    Bilinearly upsample a single [H, W] activation map to [image_size, image_size]
    and divide by its maximum value.
    """
    # F.interpolate expects [1, 1, H, W]
    amap = activation_map.detach().float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        amap,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    max_val = upsampled.max().item()
    if max_val > 0.0:
        upsampled = upsampled / max_val
    return upsampled.cpu().numpy()


def boxes_to_union_mask(
    boxes: torch.Tensor,
    image_size: int = 320,
) -> np.ndarray:
    """
    Convert normalized [N, 4] boxes ([x_min, y_min, x_max, y_max]) into a
    boolean [image_size, image_size] union mask.
    """
    mask = np.zeros((image_size, image_size), dtype=bool)
    if boxes.numel() == 0:
        return mask

    boxes_np = boxes.detach().cpu().numpy()
    for x_min, y_min, x_max, y_max in boxes_np:
        x1 = int(np.clip(np.floor(x_min * image_size), 0, image_size))
        y1 = int(np.clip(np.floor(y_min * image_size), 0, image_size))
        x2 = int(np.clip(np.ceil(x_max * image_size), 0, image_size))
        y2 = int(np.clip(np.ceil(y_max * image_size), 0, image_size))
        mask[y1:y2, x1:x2] = True
    return mask


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection over Union between two boolean masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_pointing_hit(norm_heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    """Return 1.0 if the activation maximum lies inside the reference box union, else 0.0."""
    max_y, max_x = np.unravel_index(np.argmax(norm_heatmap), norm_heatmap.shape)
    return 1.0 if gt_mask[max_y, max_x] else 0.0


def select_classification_threshold(
    val_labels: torch.Tensor,
    val_probs: torch.Tensor,
) -> tuple[float, float]:
    """
    Select the probability threshold that maximizes F1-score on validation data.
    Returns (best_threshold, best_val_f1).
    """
    y_true = val_labels.detach().cpu().numpy().astype(int)
    y_prob = val_probs.detach().cpu().numpy()

    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(thresh)

    return best_thresh, best_f1


def select_heatmap_threshold(
    val_labels: torch.Tensor,
    val_activation_maps: torch.Tensor,
    val_boxes: list[torch.Tensor],
    image_size: int = 320,
    thresholds: list[float] = HEATMAP_THRESHOLDS,
) -> tuple[float, float]:
    """
    Select the heatmap threshold from {0.1, ..., 0.9} that maximizes mean
    validation IoU on all annotated positive images.
    Returns (best_threshold, best_val_iou).
    """
    # Precompute upsampled heatmaps and GT union masks for annotated positive images
    pos_heatmaps = []
    pos_gt_masks = []
    for i in range(len(val_labels)):
        if val_labels[i].item() == 1.0 and val_boxes[i].numel() > 0:
            pos_heatmaps.append(upsample_and_normalize_map(val_activation_maps[i], image_size))
            pos_gt_masks.append(boxes_to_union_mask(val_boxes[i], image_size))

    if not pos_heatmaps:
        return 0.5, 0.0

    best_thresh = thresholds[0]
    best_mean_iou = -1.0
    for thresh in thresholds:
        ious = [
            compute_iou(hmap >= thresh, gt_mask)
            for hmap, gt_mask in zip(pos_heatmaps, pos_gt_masks)
        ]
        mean_iou = float(np.mean(ious))
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_thresh = float(thresh)

    return best_thresh, best_mean_iou


def evaluate_split(
    labels: torch.Tensor,
    probs: torch.Tensor,
    activation_maps: torch.Tensor,
    boxes: list[torch.Tensor],
    cls_threshold: float,
    heatmap_threshold: float,
    image_size: int = 320,
) -> dict[str, float]:
    """
    Evaluate classification (AUROC, F1) and localization (mean IoU, pointing-game hit rate)
    on a dataset split using fixed validation-selected thresholds.
    """
    y_true = labels.detach().cpu().numpy().astype(int)
    y_prob = probs.detach().cpu().numpy().astype(np.float64)

    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    y_pred = (y_prob >= cls_threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    ious = []
    hits = []
    for i in range(len(labels)):
        # Include ALL annotated positive images (not only correctly classified positives)
        if labels[i].item() == 1.0 and boxes[i].numel() > 0:
            hmap = upsample_and_normalize_map(activation_maps[i], image_size)
            gt_mask = boxes_to_union_mask(boxes[i], image_size)
            ious.append(compute_iou(hmap >= heatmap_threshold, gt_mask))
            hits.append(compute_pointing_hit(hmap, gt_mask))

    mean_iou = float(np.mean(ious)) if ious else float("nan")
    hit_rate = float(np.mean(hits)) if hits else float("nan")

    return {
        "auroc": auroc,
        "f1": f1,
        "mean_iou": mean_iou,
        "hit_rate": hit_rate,
        "cls_threshold": float(cls_threshold),
        "heatmap_threshold": float(heatmap_threshold),
        "num_annotated_positives": len(ious),
    }


def evaluate_run(
    val_preds: dict,
    test_preds: dict,
    image_size: int = 320,
) -> dict[str, dict[str, float]]:
    """
    End-to-end evaluation for a single training run:
      1. Select classification F1 threshold & heatmap IoU threshold on validation data.
      2. Apply those fixed thresholds to evaluate both validation and test splits.

    Expected keys in val_preds / test_preds:
      - "labels": [N] float tensor (0 or 1)
      - "probs": [N] float tensor (predicted probabilities; or "logits" which will be sigmoided)
      - "activation_maps": [N, H, W] float tensor
      - "boxes": list of N tensors, each [M_i, 4] normalized [x_min, y_min, x_max, y_max]
    """
    val_probs = val_preds["probs"] if "probs" in val_preds else torch.sigmoid(val_preds["logits"])
    test_probs = test_preds["probs"] if "probs" in test_preds else torch.sigmoid(test_preds["logits"])

    cls_thresh, _ = select_classification_threshold(val_preds["labels"], val_probs)
    hm_thresh, _ = select_heatmap_threshold(
        val_preds["labels"], val_preds["activation_maps"], val_preds["boxes"], image_size
    )

    val_metrics = evaluate_split(
        val_preds["labels"], val_probs, val_preds["activation_maps"], val_preds["boxes"],
        cls_threshold=cls_thresh, heatmap_threshold=hm_thresh, image_size=image_size,
    )
    test_metrics = evaluate_split(
        test_preds["labels"], test_probs, test_preds["activation_maps"], test_preds["boxes"],
        cls_threshold=cls_thresh, heatmap_threshold=hm_thresh, image_size=image_size,
    )
    return {"val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    # Quick synthetic test
    torch.manual_seed(0)
    N, H, W = 6, 10, 10

    def _make_synthetic_split():
        labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        probs = torch.tensor([0.90, 0.75, 0.40, 0.20, 0.15, 0.35])
        amaps = torch.full((N, H, W), 0.005)
        # Put strong activation inside [0.2, 0.2, 0.6, 0.6] for positive images
        amaps[0, 2:6, 2:6] = 0.05
        amaps[1, 3:6, 3:6] = 0.05
        amaps[2, 7:9, 7:9] = 0.05  # Deliberate miss on 3rd positive
        amaps = amaps / amaps.view(N, -1).sum(dim=-1).view(N, 1, 1)

        boxes = [
            torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
            torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
            torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
            torch.empty((0, 4)),
            torch.empty((0, 4)),
            torch.empty((0, 4)),
        ]
        return {"labels": labels, "probs": probs, "activation_maps": amaps, "boxes": boxes}

    val_data = _make_synthetic_split()
    test_data = _make_synthetic_split()
    results = evaluate_run(val_data, test_data, image_size=320)

    for split_name, metrics in results.items():
        print(
            f"{split_name.upper():4s} | AUROC={metrics['auroc']:.4f} | "
            f"F1={metrics['f1']:.4f} (thr={metrics['cls_threshold']:.2f}) | "
            f"IoU={metrics['mean_iou']:.4f} (thr={metrics['heatmap_threshold']:.1f}) | "
            f"HitRate={metrics['hit_rate']:.4f}"
        )