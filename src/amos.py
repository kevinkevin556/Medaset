import os
from typing import Literal

import numpy as np
from monai.data import CacheDataset
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.transforms import Transform as MonaiTransform

from base import BaseMixIn

abbr = {
    "train": "Tr",
    "validation": "Va",
    "test": "Ts",
}


def get_file_number(filename):
    return int(str(filename).split("_")[-1].split(".")[0])


class AMOSDataset(BaseMixIn, CacheDataset):
    def __init__(
        self,
        root_dir: str,
        modality: str = "ct",
        stage: Literal["train", "validation", "test"] = "train",
        transform: MonaiTransform = None,
        n_classes=16,
        mask_mapping: dict = None,
        dev: bool = False,
        cache_rate: float = 0.1,
        num_workers: int = 2,
    ):
        # Dataset info
        self.modality = modality
        self.stage = stage

        BaseMixIn.__init__(
            self,
            image_dir=os.path.join(root_dir, f"images{abbr[self.stage]}"),
            target_dir=os.path.join(root_dir, f"labels{abbr[self.stage]}"),
            n_classes=n_classes,
            mask_mapping=mask_mapping,
        )

        # Collect data with specified modality
        if self.modality == "ct":
            self.image_path = [p for p in self.image_path if get_file_number(p) <= 500]
            self.target_path = [p for p in self.target_path if get_file_number(p) <= 500]
        elif self.modality == "mr":
            self.image_path = [p for p in self.image_path if get_file_number(p) > 500]
            self.target_path = [p for p in self.target_path if get_file_number(p) > 500]
        elif self.modality == "ct+mr":
            pass
        else:
            raise ValueError("Invalid modality is specified. Options are {ct, mr, ct+mr}.")

        # Developing mode (20% for training data, 5% for other stage)
        if dev:
            if stage == "train":
                n_train_dev = max(int(len(self.image_path) * 0.2), 40)
                self.image_path = self.image_path[:n_train_dev]
                self.target_path = self.target_path[:n_train_dev]
            else:
                n_val_dev = max(int(len(self.image_path) * 0.02), 5)
                self.image_path = self.image_path[:n_val_dev]
                self.target_path = self.target_path[:n_val_dev]

        # Transformation
        if isinstance(transform, MonaiTransform):
            pass
        elif stage == "train":
            transform = amos_train_transforms
        elif (stage == "validation") or (stage == "test"):
            transform = amos_val_transforms
        else:
            raise ValueError("Either stage or transform should be specified.")

        # Initialize CacheDataset
        CacheDataset.__init__(
            self,
            data=[{"image": im, "label": la} for im, la in zip(self.image_path, self.target_path)],
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self.target_path)

    def __getitem__(self, index):
        batch = CacheDataset.__getitem__(self, index)
        if self.mask_mapping is not None:
            if isinstance(batch, dict):
                for ori_val, new_val in self.mask_mapping.items():
                    batch["label"][batch["label"] == ori_val] = new_val
            elif isinstance(batch, list):
                for i in range(len(batch)):
                    _b = batch[i]
                    for ori_val, new_val in self.mask_mapping.items():
                        _b["label"][_b["label"] == ori_val] = new_val
                    batch[i] = _b
            else:
                raise TypeError("Data type other than dict or list can not modify the label part through mask_mapping.")
        return batch


amos_train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=275,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=1.0,
            spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi / 30),
            scale_range=(0.1, 0.1, 0.1),
        ),
        ToTensord(keys=["image", "label"]),
    ]
)


amos_val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=275,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)
