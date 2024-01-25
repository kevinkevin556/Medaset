from __future__ import annotations

from collections.abc import Sequence, tuple
from pathlib import Path
from typing import Any

import cv2
from monai.config import PathLike
from monai.data import ImageReader
from numpy import ndarray


class CV2Reader(ImageReader):
    def __init__(self, flags):
        self.flags = flags

    def get_data(self, img) -> tuple[ndarray, dict]:
        return img, {}

    def read(self, data: PathLike + Sequence[PathLike], **kwargs) -> Sequence[Any] | Any:
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

    def verify_suffix(self, filename: PathLike | Sequence[PathLike]) -> bool:
        if isinstance(filename, (list, tuple)):
            return all([Path(f).suffix in (".png", "jpg") for f in filename])
        else:
            return Path(filename).suffix in (".png", ".jpg")
