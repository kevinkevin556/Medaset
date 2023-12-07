from pathlib import Path
from typing import Any, Sequence, Tuple, Union

import cv2
import monai
import numpy as np
from monai.config import PathLike
from monai.data import ImageReader, PydicomReader
from monai.data.image_reader import _stack_images
from monai.utils import (
    MetaKeys,
    SpaceKeys,
    TraceKeys,
    ensure_tuple,
    optional_import,
    require_pkg,
)
from numpy import ndarray


class CV2Reader(ImageReader):
    def __init__(self, flags):
        self.flags = flags

    def get_data(self, img) -> Tuple[ndarray, dict]:
        return img, {}

    def read(self, data: Union[PathLike, Sequence[PathLike]], **kwargs) -> Union[Sequence[Any], Any]:
        if not isinstance(data, (list, tuple)):
            data = [data]

        output = []
        for filename in data:
            img = cv2.imread(filename, flags=self.flags)
            if img is not None:
                output.append(img)
            else:
                raise RuntimeError(f"Image can not be read from the path {filename}")

        return output if len(output) > 1 else output[0]

    def verify_suffix(self, filename: Union[PathLike, Sequence[PathLike]]) -> bool:
        if isinstance(filename, (list, tuple)):
            return all([Path(f).suffix in (".png", "jpg") for f in filename])
        else:
            return Path(filename).suffix in (".png", ".jpg")


class IncompatiblePydicomReader(PydicomReader):
    def get_data(self, data) -> Tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            data: a pydicom dataset object, or a list of pydicom dataset objects, or a list of list of
                pydicom dataset objects.

        """

        dicom_data = []
        # combine dicom series if exists
        if self.has_series is True:
            # a list, all objects within a list belong to one dicom series
            if not isinstance(data[0], list):
                dicom_data.append(self._combine_dicom_series(data))
            # a list of list, each inner list represents a dicom series
            else:
                for series in data:
                    dicom_data.append(self._combine_dicom_series(series))
        else:
            # a single pydicom dataset object
            if not isinstance(data, list):
                data = [data]
            for d in data:
                if hasattr(d, "SegmentSequence"):
                    data_array, metadata = self._get_seg_data(d)
                else:
                    data_array = self._get_array_data(d)
                    metadata = self._get_meta_dict(d)
                    metadata[MetaKeys.SPATIAL_SHAPE] = data_array.shape
                dicom_data.append((data_array, metadata))

        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}

        for data_array, metadata in ensure_tuple(dicom_data):
            img_array.append(np.ascontiguousarray(np.swapaxes(data_array, 0, 1) if self.swap_ij else data_array))

        return _stack_images(img_array, compatible_meta), compatible_meta
