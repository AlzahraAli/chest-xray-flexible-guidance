
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA_DIR = Path.home() / "cs421-data"
OUTPUT_DIR = DATA_DIR / "splits"
SEED = 42

# Load annotation and metadata files
labels = pd.read_csv(DATA_DIR / "stage_2_train_labels.csv")

nih = pd.read_csv(
    DATA_DIR / "Data_Entry_2017_v2020.csv",
    usecols=["Image Index", "Patient ID"]
)

with open(
    DATA_DIR / "pneumonia-challenge-dataset-mappings_2018.json",
    encoding="utf-8"
) as file:
    mapping = pd.DataFrame(json.load(file))

mapping = mapping[["subset_img_id", "img_id"]]

mapping = mapping.rename(columns={
    "subset_img_id": "rsna_id",
    "img_id": "nih_image"
})

nih = nih.rename(columns={
    "Image Index": "nih_image",
    "Patient ID": "nih_patient_id"
})

# Combine annotations belonging to the same image
records = []

for image_id, group in labels.groupby("patientId"):
    targets = group["Target"].unique()

    if len(targets) != 1 or targets[0] not in (0, 1):
        raise ValueError(f"Invalid labels: {image_id}")

    target = int(targets[0])
    coordinates = group[["x", "y", "width", "height"]]

    if target == 1:
        if coordinates.isna().any().any():
            raise ValueError(f"Missing positive box: {image_id}")

        if (coordinates[["width", "height"]] <= 0).any().any():
            raise ValueError(f"Invalid box size: {image_id}")

        boxes = coordinates.values.tolist()
    else:
        if coordinates.notna().any().any():
            raise ValueError(f"Unexpected negative box: {image_id}")

        boxes = []

    records.append({
        "rsna_id": image_id,
        "target": target,
        "boxes": json.dumps(boxes)
    })

data = pd.DataFrame(records)

print("Original images:", len(data))

# Recover the original NIH patient identifier
data = data.merge(
    mapping,
    on="rsna_id",
    how="left",
    validate="one_to_one"
)

data = data.merge(
    nih,
    on="nih_image",
    how="left",
    validate="many_to_one"
)

missing = data["nih_patient_id"].isna().sum()
print("Images without verified patient IDs:", missing)

data = data.dropna(subset=["nih_patient_id"]).copy()
data["nih_patient_id"] = data["nih_patient_id"].astype(int)

# Verify that every retained image exists
image_ids = {
    path.stem
    for path in (DATA_DIR / "stage_2_train_images").glob("*.dcm")
}

missing_images = ~data["rsna_id"].isin(image_ids)

if missing_images.any():
    raise FileNotFoundError(
        f"{missing_images.sum()} DICOM images are missing."
    )

print("Retained images:", len(data))
print("Unique patients:", data["nih_patient_id"].nunique())
print("Positive images:", int(data["target"].sum()))
print("Negative images:", int((data["target"] == 0).sum()))

# Allocate approximately 80% of patients to training
first_split = GroupShuffleSplit(
    n_splits=1,
    train_size=0.8,
    random_state=SEED
)

train_idx, remaining_idx = next(
    first_split.split(data, groups=data["nih_patient_id"])
)

train = data.iloc[train_idx].copy()
remaining = data.iloc[remaining_idx].copy()

# Divide the remaining patients between validation and testing
second_split = GroupShuffleSplit(
    n_splits=1,
    train_size=0.5,
    random_state=SEED
)

val_idx, test_idx = next(
    second_split.split(
        remaining,
        groups=remaining["nih_patient_id"]
    )
)

val = remaining.iloc[val_idx].copy()
test = remaining.iloc[test_idx].copy()

# Check for patient overlap
train_patients = set(train["nih_patient_id"])
val_patients = set(val["nih_patient_id"])
test_patients = set(test["nih_patient_id"])

if (
    train_patients & val_patients
    or train_patients & test_patients
    or val_patients & test_patients
):
    raise ValueError("Patient overlap detected.")

if len(train) + len(val) + len(test) != len(data):
    raise ValueError("Split counts do not match.")

print("\nPatient overlap: 0")

# Save the fixed dataset splits
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary = []

for name, split in [
    ("train", train),
    ("val", val),
    ("test", test)
]:
    if split["target"].nunique() != 2:
        raise ValueError(f"Both classes are not present in {name}.")

    split.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    summary.append({
        "split": name,
        "images": len(split),
        "patients": split["nih_patient_id"].nunique(),
        "positive": int(split["target"].sum()),
        "negative": int((split["target"] == 0).sum()),
        "image_percentage": round(100 * len(split) / len(data), 2)
    })

summary = pd.DataFrame(summary)
summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

print("\nDataset split summary:")
print(summary.to_string(index=False))
print("\nFiles saved to:", OUTPUT_DIR)
