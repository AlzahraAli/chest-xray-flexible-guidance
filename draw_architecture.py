from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


output = Path("evidence/figures")
output.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 8.5))

ax.set_xlim(0, 12)
ax.set_ylim(0, 8.5)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, text, fontsize=9):
    rect = Rectangle(
        (x, y), w, h,
        facecolor="white",
        edgecolor="black",
        linewidth=1.1
    )

    ax.add_patch(rect)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )


def arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "linewidth": 1.1,
            "shrinkA": 0,
            "shrinkB": 0
        }
    )


def section(y, title):
    ax.text(
        0.3, y,
        title,
        fontsize=11,
        fontweight="bold",
        va="center"
    )


# A. Shared model architecture

section(8.15, "A. Image processing and shared model")

box(
    0.3, 6.75, 2.0, 0.9,
    "RSNA chest X-rays\nDICOM images"
)

box(
    2.65, 6.75, 2.0, 0.9,
    "Preprocessing\n320 x 320 x 3\nImageNet normalization"
)

box(
    5.0, 6.75, 2.0, 0.9,
    "ImageNet-pretrained\nDenseNet-121"
)

box(
    7.35, 6.75, 1.95, 0.9,
    "Convolutional features\n1024 x 10 x 10"
)

box(
    9.65, 7.05, 2.05, 1.0,
    "Global average pooling\n+ linear classifier\nClassification logit"
)

box(
    9.65, 5.75, 2.05, 1.0,
    "Weighted spatial scores\n+ spatial softmax\n10 x 10 activation map"
)

arrow(2.3, 7.2, 2.65, 7.2)
arrow(4.65, 7.2, 5.0, 7.2)
arrow(7.0, 7.2, 7.35, 7.2)

arrow(9.3, 7.2, 9.65, 7.55)
arrow(9.3, 7.2, 9.65, 6.25)

ax.text(
    8.0, 5.85,
    "Shared classifier weights",
    fontsize=8,
    ha="center"
)

ax.plot(
    [0.3, 11.7],
    [5.4, 5.4],
    color="black",
    linewidth=0.6
)


# B. Training pipeline

section(5.1, "B. Training objectives and weight updates")

box(
    0.3, 3.85, 2.3, 0.8,
    "Image labels\n+ classification logits"
)

box(
    0.3, 2.65, 2.3, 0.9,
    "Training bounding boxes\n+ controlled perturbations"
)

box(
    3.0, 3.85, 2.4, 0.8,
    "Binary cross-entropy\nClassification loss"
)

box(
    3.0, 2.65, 2.4, 0.9,
    "Activation map + spatial cost\nBinary / fixed padding /\ndistance-weighted"
)

box(
    5.9, 3.3, 2.5, 1.1,
    "Combined training loss\n"
    r"$L_{\mathrm{cls}} + \lambda L_{\mathrm{spatial}}$"
    "\nSpatial loss: annotated positives only",
    fontsize=8
)

box(
    8.9, 3.3, 2.8, 1.1,
    "Backpropagation + Adam\nUpdate shared model weights\nSave best validation checkpoint"
)

arrow(2.6, 4.25, 3.0, 4.25)
arrow(2.6, 3.1, 3.0, 3.1)

arrow(5.4, 4.25, 5.9, 4.0)
arrow(5.4, 3.1, 5.9, 3.65)

arrow(8.4, 3.85, 8.9, 3.85)

ax.text(
    6.0, 2.3,
    "Classification-only baseline: spatial loss disabled",
    fontsize=9,
    ha="center"
)

ax.plot(
    [0.3, 11.7],
    [2.05, 2.05],
    color="black",
    linewidth=0.6
)


# C. Evaluation pipeline

section(1.75, "C. Validation and held-out test evaluation")

box(
    0.3, 0.45, 2.4, 0.9,
    "Load best checkpoint\nGenerate predictions\nand activation maps"
)

box(
    3.1, 0.45, 2.4, 0.9,
    "Validation set\nSelect classification\nand heatmap thresholds"
)

box(
    5.9, 0.45, 2.4, 0.9,
    "Held-out test set\nFixed thresholds\nOriginal reference boxes"
)

box(
    8.7, 0.45, 3.0, 0.9,
    "AUROC and F1-score\nMean IoU and pointing-game hit rate\nMean and SD across three seeds",
    fontsize=8
)

arrow(2.7, 0.9, 3.1, 0.9)
arrow(5.5, 0.9, 5.9, 0.9)
arrow(8.3, 0.9, 8.7, 0.9)


fig.tight_layout()

destination = output / "system_architecture.png"

fig.savefig(
    destination,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15,
    facecolor="white"
)

plt.close(fig)

print("Architecture diagram saved:", destination.resolve())
print("Export resolution: 300 DPI")