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
