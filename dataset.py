
import json
from pathlib import Path

import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


DATA_DIR = Path.home() / "cs421-data"
IMAGE_SIZE = 320

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class RSNADataset(Dataset):

    def __init__(self, split, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "stage_2_train_images"

        self.records = pd.read_csv(
            self.data_dir / "splits" / f"{split}.csv"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records.iloc[index]
        image_id = record["rsna_id"]

        # Read DICOM pixels
        image_path = self.image_dir / f"{image_id}.dcm"
        dicom = pydicom.dcmread(image_path)
        pixels = dicom.pixel_array

        if pixels.ndim != 2:
            raise ValueError(f"Expected grayscale image: {image_id}")

        height, width = pixels.shape

        if dicom.BitsStored != 8 or dicom.PixelRepresentation != 0:
            raise ValueError(f"Unexpected pixel format: {image_id}")

        image = torch.from_numpy(pixels.astype("float32")) / 255.0

        # Correct grayscale interpretation if required
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = 1.0 - image
        elif dicom.PhotometricInterpretation != "MONOCHROME2":
            raise ValueError(f"Unexpected photometric format: {image_id}")

        # Resize and convert to three channels
        image = image.unsqueeze(0)

        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )

        image = image.repeat(3, 1, 1)
        image = TF.normalize(image, mean=MEAN, std=STD)

        # Convert boxes from xywh to normalized xyxy
        boxes = torch.tensor(
            json.loads(record["boxes"]),
            dtype=torch.float32
        ).reshape(-1, 4)

        if len(boxes) > 0:
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            boxes[:, [0, 2]] /= width
            boxes[:, [1, 3]] /= height

            if (boxes < 0).any() or (boxes > 1).any():
                raise ValueError(f"Box outside image: {image_id}")

        target = torch.tensor(
            record["target"],
            dtype=torch.float32
        )

        return image, target, boxes, image_id


def collate_fn(batch):
    images, targets, boxes, image_ids = zip(*batch)

    return (
        torch.stack(images),
        torch.stack(targets),
        list(boxes),
        list(image_ids)
    )


if __name__ == "__main__":
    dataset = RSNADataset("train")

    print("Training dataset size:", len(dataset))

    for target_value in [0, 1]:
        index = dataset.records.index[
            dataset.records["target"] == target_value
        ][0]

        image, target, boxes, image_id = dataset[index]

        print("\nImage ID:", image_id)
        print("Image shape:", image.shape)
        print("Target:", target.item())
        print("Boxes shape:", boxes.shape)
        print("Boxes:", boxes)
        print("Finite pixels:", torch.isfinite(image).all().item())

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    images, targets, boxes, image_ids = next(iter(loader))

    print("\nBatch image shape:", images.shape)
    print("Batch target shape:", targets.shape)
    print("Number of box tensors:", len(boxes))
