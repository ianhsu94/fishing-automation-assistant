from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """以位元組解碼，避開 Windows OpenCV 對 Unicode 路徑的限制。"""
    path_obj = Path(path)
    try:
        encoded = np.frombuffer(path_obj.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """先在記憶體編碼再寫檔，可安全寫入中文 Windows 路徑。"""
    path_obj = Path(path)
    extension = path_obj.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True
