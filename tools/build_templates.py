from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from image_io import read_image, write_image  # noqa: E402


CROPS = {
    "title.png": ("01_ready.jpeg", (220, 75, 545, 205)),
    "bobber.png": ("04_timing_left.jpeg", (260, 1000, 330, 1130)),
    "result_banner.png": ("08_result.jpeg", (155, 310, 560, 535)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="從參考截圖重建 OpenCV 模板")
    parser.add_argument("--samples", default=str(Path(__file__).resolve().parents[1] / "samples"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "templates"))
    args = parser.parse_args()

    samples = Path(args.samples).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for output_name, (source_name, (x1, y1, x2, y2)) in CROPS.items():
        image = read_image(samples / source_name, cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"找不到或無法讀取：{samples / source_name}")
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or not write_image(output / output_name, crop):
            raise SystemExit(f"無法建立：{output / output_name}")
        print(output / output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
