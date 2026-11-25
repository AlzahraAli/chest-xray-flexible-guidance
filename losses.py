import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_normalized_grid(height: int, width: int, device: torch.device, dtype: torch.dtype):
    """Return cell-center coordinates (grid_x, grid_y) in normalized [0, 1] image space."""
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y  # Each has shape [H, W]


def _min_squared_distance_to_union(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute squared Euclidean distance from each grid cell center to the nearest
    point in the union of boxes B (0.0 inside any box).

    Args:
        grid_x, grid_y: [H, W] normalized cell centers.
        boxes: [N, 4] normalized [x_min, y_min, x_max, y_max], N >= 1.

    Returns:
        [H, W] tensor of squared distances d(p, B)^2.
    """
    # Broadcast grid to [1, H, W] and box edges to [N, 1, 1]
    gx = grid_x.unsqueeze(0)
    gy = grid_y.unsqueeze(0)

    x_min = boxes[:, 0].view(-1, 1, 1)
    y_min = boxes[:, 1].view(-1, 1, 1)
    x_max = boxes[:, 2].view(-1, 1, 1)
    y_max = boxes[:, 3].view(-1, 1, 1)

    dx = torch.clamp(x_min - gx, min=0.0) + torch.clamp(gx - x_max, min=0.0)
    dy = torch.clamp(y_min - gy, min=0.0) + torch.clamp(gy - y_max, min=0.0)

    dist_sq_per_box = dx * dx + dy * dy  # [N, H, W]
    return dist_sq_per_box.min(dim=0).values  # [H, W]


def _pad_boxes(boxes: torch.Tensor, pad_frac: float = 0.10) -> torch.Tensor:
    """Expand each box by pad_frac of its width and height on each side, clipped to [0, 1]."""
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x_max - x_min
    h = y_max - y_min
    padded = torch.stack(
        [
            x_min - pad_frac * w,
            y_min - pad_frac * h,
            x_max + pad_frac * w,
            y_max + pad_frac * h,
        ],
        dim=1,
    )
    return torch.clamp(padded, 0.0, 1.0)


class GuidedClassificationLoss(nn.Module):
    """
    Combined binary cross-entropy and spatial guidance loss (Equations 2, 3, 4).
    Matches the shared interface:
        logits: [B]
        activation_map: [B, H, W] (spatial softmax distribution, sums to 1 per image)
        labels: [B] float tensor in {0.0, 1.0}
        boxes: list of B tensors, each [N_i, 4] normalized [x_min, y_min, x_max, y_max]
    """

    def __init__(
        self,
        method: str = "distance_weighted",
        lam: float = 1.0,
        rho: float = 0.05,
        pad_frac: float = 0.10,
    ):
        super().__init__()
        valid_methods = {"cls_only", "binary", "fixed_padding", "distance_weighted"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        self.method = method
        self.lam = 0.0 if method == "cls_only" else float(lam)
        self.rho = float(rho)
        self.pad_frac = float(pad_frac)

    def compute_cost_map(
        self,
        boxes: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute the [H, W] spatial cost map for a single annotated image."""
        grid_x, grid_y = _make_normalized_grid(height, width, device, dtype)

        if self.method == "binary":
            dist_sq = _min_squared_distance_to_union(grid_x, grid_y, boxes)
            return (dist_sq > 0.0).to(dtype)

        if self.method == "fixed_padding":
            padded_boxes = _pad_boxes(boxes, pad_frac=self.pad_frac)
            dist_sq = _min_squared_distance_to_union(grid_x, grid_y, padded_boxes)
            return (dist_sq > 0.0).to(dtype)

        if self.method == "distance_weighted":
            dist_sq = _min_squared_distance_to_union(grid_x, grid_y, boxes)
            return 1.0 - torch.exp(-dist_sq / (self.rho ** 2))

        return torch.zeros((height, width), device=device, dtype=dtype)

    def forward(
        self,
        logits: torch.Tensor,
        activation_map: torch.Tensor,
        labels: torch.Tensor,
        boxes: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size, height, width = activation_map.shape
        device = activation_map.device
        dtype = activation_map.dtype

        cls_per_img = F.binary_cross_entropy_with_logits(
            logits, labels.to(dtype), reduction="none"
        )
        spatial_per_img = torch.zeros(batch_size, device=device, dtype=dtype)

        if self.method != "cls_only" and self.lam > 0.0:
            for i in range(batch_size):
                # Apply spatial penalty ONLY when y == 1 and box set B is non-empty
                if labels[i].item() == 1.0 and boxes[i].numel() > 0:
                    img_boxes = boxes[i].to(device=device, dtype=dtype)
                    cost_map = self.compute_cost_map(img_boxes, height, width, device, dtype)
                    spatial_per_img[i] = (activation_map[i] * cost_map).sum()

        total_per_img = cls_per_img + self.lam * spatial_per_img
        return {
            "total": total_per_img.mean(),
            "cls": cls_per_img.mean(),
            "spatial": spatial_per_img.mean(),
        }


if __name__ == "__main__":
    # Quick synthetic test matching teammate's exact interface
    B, H, W = 2, 10, 10
    logits = torch.tensor([0.8, -0.5], dtype=torch.float32, requires_grad=True)
    raw_scores = torch.randn(B, H, W, requires_grad=True)
    activation_map = F.softmax(raw_scores.view(B, -1), dim=-1).view(B, H, W)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    boxes = [
        torch.tensor([[0.2, 0.2, 0.6, 0.6]], dtype=torch.float32),  # Positive image
        torch.empty((0, 4), dtype=torch.float32),                   # Negative image
    ]

    for method in ["cls_only", "binary", "fixed_padding", "distance_weighted"]:
        criterion = GuidedClassificationLoss(method=method, lam=1.0, rho=0.05)
        out = criterion(logits, activation_map, labels, boxes)
        print(
            f"{method:18s} | total={out['total'].item():.4f} "
            f"cls={out['cls'].item():.4f} spatial={out['spatial'].item():.4f}"
        )