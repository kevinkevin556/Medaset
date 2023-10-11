import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

from .utils import apply_window, get_file_paths, get_paths_from_json, read_image


class BaseMixIn:
    def __init__(
        self,
        image_dir: Optional[Sequence] = None,
        target_dir: Optional[Sequence] = None,
        root_dir: Optional[str] = None,
        dataset_json: Optional[Sequence] = None,
        num_classes: Optional[int] = -1,
        cmap: Optional[Union[str, ListedColormap]] = None,
        mask_mapping: Optional[dict] = None,
    ):
        assert (image_dir is not None) or (dataset_json is not None), "Either image_dir"
        if image_dir is not None:
            self.image_path = get_file_paths(image_dir)
            self.target_path = get_file_paths(target_dir) if target_dir else [None] * len(self.image_path)
        else:
            self.image_path, self.target_path = get_paths_from_json(dataset_json, root_dir=root_dir)

        self.mask_mapping = mask_mapping
        self.num_classes = num_classes if num_classes else len(mask_mapping)
        assert self.num_classes >= 2, "The number of classes is at least 2."

        # Colormap
        self.cmap = cmap
        if self.cmap is None:
            if self.num_classes == 2:  # Default binary colormap
                self.cmap = "gray"
            else:  # Default multiclass colormap
                cmap = plt.get_cmap("rainbow")(np.linspace(0, 1, self.num_classes - 1))
                cmap = np.insert(cmap, 0, [0, 0, 0, 1], axis=0)
                self.cmap = ListedColormap(cmap)

    def plot(
        self,
        index: int,
        figsize: Union[int, tuple] = (6, 3),
        dpi: int = 100,
        window: Optional[tuple] = None,
        cmap: Union[str, ListedColormap] = None,
    ):
        sample = self[index]
        if isinstance(sample, tuple):
            image, label = sample
        elif isinstance(sample, dict):
            image, label = sample["image"], sample["label"]
        else:
            raise ValueError("Invalid type to be unpacked into image and label.")

        image = apply_window(image, *window) if window else image
        cmap = self.cmap if cmap is None else cmap
        figsize = (figsize, figsize // 2) if isinstance(figsize, int) else figsize

        fig, ax = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0, "hspace": 0}, dpi=dpi)
        ax[0].imshow(to_pil_image(image), cmap="gray")
        ax[0].set_xlabel("Image")
        ax[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[1].imshow(to_pil_image(label.byte()), cmap=cmap)
        ax[1].set_xlabel("Ground Truth")
        ax[1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        fig.suptitle(Path(self.image_path[index]).name)
        return fig, ax

    def nnunet_format(
        self, id, name, channel_names, labels={"background": 0, "1": 1}, root_dir="./data/nnUNet/nnUNet_raw"
    ):
        dataset_json = {
            "name": name,
            "channel_names": channel_names,
            "labels": labels,
            "numTraining": len(self),
            "file_ending": ".nii.gz",
            "training": [],
        }

        # Create destination directory
        src_dir = Path(root_dir) / f"Dataset{str(id).rjust(3, '0')}_{name}"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (src_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

        # Create nii files and write file path into dataset_jon
        n_digits = len(str(len(self)))
        for i, (image_path, label_path) in enumerate(zip(self.image_path, self.target_path)):
            image = read_image(image_path).astype(np.int16)
            image_save_paths = []
            for c in range(image.shape[-1]):
                image_c = nib.Nifti1Image(image[:, :, c], np.eye(4))
                image_c_name = f"{name}_{str(i).rjust(n_digits, '0')}_000{c}.nii.gz"
                image_save_paths.append(str(src_dir / "imagesTr" / image_c_name))
                nib.save(image_c, image_save_paths[-1])

            label = read_image(label_path, mask_mapping=self.mask_mapping).astype(np.int16)
            label = nib.Nifti1Image(label, np.eye(4))
            label_name = f"{name}_{str(i).rjust(n_digits, '0')}.nii.gz"
            label_save_path = src_dir / "labelsTr" / label_name
            nib.save(label, label_save_path)

            if len(image_save_paths) == 1:
                dataset_json["training"].append({"image": image_save_paths[0], "label": str(label_save_path)})
            else:
                dataset_json["training"].append({"image": image_save_paths, "label": str(label_save_path)})

        # Generate dataset.json
        with open(Path(src_dir / "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=4)
