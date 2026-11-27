
import numpy as np
import torch


def perturb_boxes(
    boxes: torch.Tensor,
    condition: str = "unchanged",
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Apply a controlled perturbation to normalized bounding boxes [x_min, y_min, x_max, y_max].

    Args:
        boxes: FloatTensor of shape [N, 4] in normalized [0, 1] coordinates.
               Negative images pass an empty tensor of shape [0, 4].
        condition: One of "unchanged", "translate", "shrink", "enlarge".
        rng: Optional NumPy random Generator for reproducible translations.

    Returns:
        FloatTensor of shape [N, 4], clipped to [0.0, 1.0].
    """
    if boxes.numel() == 0:
        return boxes.clone()

    if condition == "unchanged":
        return torch.clamp(boxes.clone(), 0.0, 1.0)

    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x_max - x_min
    h = y_max - y_min

    if condition == "translate":
        if rng is None:
            rng = np.random.default_rng()
        # Random shift up to +/- 10% of each box's width and height
        n = boxes.shape[0]
        dx = torch.tensor(rng.uniform(-0.10, 0.10, size=n), dtype=boxes.dtype, device=boxes.device) * w
        dy = torch.tensor(rng.uniform(-0.10, 0.10, size=n), dtype=boxes.dtype, device=boxes.device) * h
        new_boxes = torch.stack([x_min + dx, y_min + dy, x_max + dx, y_max + dy], dim=1)

    elif condition == "shrink":
        # Centered 20% shrink -> scale = 0.8
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        half_w = 0.5 * (0.80 * w)
        half_h = 0.5 * (0.80 * h)
        new_boxes = torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)

    elif condition == "enlarge":
        # Centered 20% enlargement -> scale = 1.2
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        half_w = 0.5 * (1.20 * w)
        half_h = 0.5 * (1.20 * h)
        new_boxes = torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)

    else:
        raise ValueError(f"Unknown perturbation condition: {condition}")

    return torch.clamp(new_boxes, 0.0, 1.0)

def perturb_dataset_boxes(
    boxes_by_id: dict[str, torch.Tensor],
    condition: str = "unchanged",
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Generate a reproducible perturbed box dict once per run to share across guided methods."""
    rng = np.random.default_rng(seed)
    return {
        img_id: perturb_boxes(boxes, condition=condition, rng=rng)
        for img_id, boxes in sorted(boxes_by_id.items())
    }

if __name__ == "__main__":
    # Quick synthetic test
    sample_boxes = torch.tensor([[0.20, 0.30, 0.60, 0.70]], dtype=torch.float32)
    rng = np.random.default_rng(42)
    for cond in ["unchanged", "translate", "shrink", "enlarge"]:
        out = perturb_boxes(sample_boxes, condition=cond, rng=rng)
        print(f"{cond:10s} -> {out.tolist()}")

    # Test empty tensor (negative image)
    empty = torch.empty((0, 4), dtype=torch.float32)
    print("empty      ->", perturb_boxes(empty, "shrink").shape)