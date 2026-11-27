import json
from pathlib import Path

import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

from PIL import Image
from run_evaluation import sha256, verify_split_hash, primary_runs
from visualize import plot_heatmap_comparison


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_predictions(root, name):
    folder = root / "evaluation" / name

    meta = json.loads(
        (folder / "metrics.json").read_text(encoding="utf-8")
    )

    path = folder / "predictions.pt"

    require(
        meta["run"] == name and meta["seed"] == 42,
        f"{name}: wrong metadata"
    )

    require(
        sha256(path) == meta["predictions_sha256"],
        f"{name}: prediction hash mismatch"
    )

    predictions = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    return meta, predictions["test"]


def select_examples(preds):
    ids = preds["image_ids"]

    positive = sorted(
        (image_id, i)
        for i, image_id in enumerate(ids)
        if preds["labels"][i].item() == 1
    )

    single = next(
        (i for _, i in positive if len(preds["boxes"][i]) == 1),
        None
    )

    multiple = next(
        (i for _, i in positive if len(preds["boxes"][i]) > 1),
        None
    )

    require(
        single is not None and multiple is not None,
        "Missing example category"
    )

    return [
        ("single_box", single),
        ("multiple_boxes", multiple)
    ]


def main():
    from dataset import DATA_DIR, RSNADataset

    root = Path(DATA_DIR)
    test_hash = verify_split_hash(root, "test")

    dataset = RSNADataset("test", data_dir=root)

    meta, baseline = load_predictions(root, "baseline_seed_42")

    ids = baseline["image_ids"]

    require(
        ids == dataset.records["rsna_id"].tolist(),
        "Dataset order mismatch"
    )

    require(
        len(ids) == 2600 and len(set(ids)) == 2600,
        "Unexpected test image count"
    )

    chosen = select_examples(baseline)

    runs = [
        r for r in primary_runs()
        if r[3] == 42 and r[2] in ("unchanged", "translate")
    ]

    require(len(runs) == 7, "Unexpected figure run count")

    out = root / "final_results" / "heatmaps"

    require(out.parent.is_dir(), "Missing final_results")

    out.mkdir(exist_ok=False)

    rows = []
    cache = {}

    for category, i in chosen:
        image, label, boxes, image_id = dataset[i]

        require(
            image_id == ids[i] and label.item() == 1,
            "Image selection mismatch"
        )

        require(
            torch.equal(boxes, baseline["boxes"][i]),
            "Original box mismatch"
        )

        cache[category] = (i, image, boxes, image_id)

        print(
            f"Selected {category}: {image_id} ({len(boxes)} boxes)",
            flush=True
        )

    for name, method, condition, seed in runs:

        meta, preds = load_predictions(root, name)

        require(
            meta["method"] == method
            and meta["condition"] == condition,
            f"{name}: incorrect method/condition"
        )

        require(
            meta["split_sha256"]["test"] == test_hash,
            f"{name}: split hash mismatch"
        )

        require(
            preds["image_ids"] == ids
            and torch.equal(preds["labels"], baseline["labels"]),
            f"{name}: image or label mismatch"
        )

        for category, (i, image, boxes, image_id) in cache.items():

            require(
                torch.equal(preds["boxes"][i], boxes),
                f"{name}: reference box mismatch"
            )

            amap = preds["activation_maps"][i]

            require(
                tuple(amap.shape) == (10, 10)
                and torch.isfinite(amap).all().item()
                and (amap >= 0).all().item()
                and abs(amap.sum().item() - 1) < 1e-5,
                f"{name}: invalid map"
            )

            path = out / f"{category}_{method}_{condition}.png"

            title = (
                f'{method.replace("_", " ").title()} '
                f"| {condition} | Seed {seed}"
            )

            plot_heatmap_comparison(
                image,
                amap,
                boxes,
                path,
                title=title,
                imagenet_normalized=True,
                dpi=300
            )

            with Image.open(path) as png:
                require(
                    all(
                        abs(d - 300) < 1
                        for d in png.info.get("dpi", (0, 0))
                    ),
                    f"{path}: incorrect DPI"
                )

            rows.append({
                "example": category,
                "image_id": image_id,
                "boxes": len(boxes),
                "run": name,
                "method": method,
                "condition": condition,
                "seed": seed,
                "probability": float(preds["probs"][i]),
                "file": path.name,
                "figure_sha256": sha256(path)
            })

    require(len(rows) == 14, "Unexpected figure count")

    pd.DataFrame(rows).to_csv(
        out / "heatmap_manifest.csv",
        index=False
    )

    print(
        "HEATMAP GENERATION PASSED: "
        "14 figures, 2 matched test images, 7 models."
    )

    print("Saved results:", out)
    print("No model inference or training performed.")


if __name__ == "__main__":
    main()