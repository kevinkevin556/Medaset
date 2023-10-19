from typing import Dict, Hashable, Mapping, Optional

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.transforms import MapTransform, Transform
from monai.utils import convert_to_tensor

__all__ = [
    "ApplyMaskMapping",
    "ApplyMaskMappingd",
    "AddBackground",
]


class ApplyMaskMapping(Transform):
    def __init__(self, mask_mapping: Optional[dict] = None) -> None:
        self.mask_mapping = mask_mapping

    def __call__(self, mask) -> torch.Tensor:
        if self.mask_mapping:
            for original_value, new_value in self.mask_mapping.items():
                mask[mask == original_value] = new_value
        return mask


class ApplyMaskMappingd(MapTransform):
    """
    Dictionary-based version `ApplyMaskMapping`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_mapping: dict,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.mask_mapper = ApplyMaskMapping(mask_mapping)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.mask_mapper(d[key])
        return d


class BackgroundifyClasses(Transform):
    """
    Modifies ground truth masks by assigning a value of zero to specific classes,
    merging them into the background class.
    """

    def __init__(self, classes: list, dim=0) -> None:
        self.classes = torch.tensor(classes)
        self.dim = dim

    def __call__(self, img) -> torch.Tensor:
        ch = img.shape[self.dim]
        # Create indexing tuple to retrieve background anc the merging class
        # E.g. ix_bg = (0, slice(None), slice(None)) for the background of image  = img[0, :, :] when dim = 0
        #      ix_cls = (slice(None), self.classes, slice(None), slice(None))
        #      for the merging classes = img[:, classes, :, :] when dim = 1
        ix_bg = tuple([0 if (d == self.dim) else slice(None) for d in range(len(img.shape))])
        ix_cls = tuple([self.classes if (d == self.dim) else slice(None) for d in range(len(img.shape))])
        img_t = torch.Tensor(img)
        # If ch == 1, the input tensor stores labels.
        # Otherwise, the input tensor is a tensor of activations (probabilities) or results from one-hot encoding.
        if ch == 1:
            img_t[torch.isin(img_t, self.classes)] = 0
        else:
            img_t[ix_bg] += torch.sum(img_t[ix_cls], dim=self.dim)
            img_t[ix_cls] = 0
        return img_t


class BackgroundifyClassesd(MapTransform):
    """
    Dictionary-based version `BackgroundifyClasses`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        classes: list,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.backgroundifier = BackgroundifyClasses(classes)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.backgroundifier(d[key])
        return d
