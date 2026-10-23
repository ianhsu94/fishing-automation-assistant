from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from config import load_config  # noqa: E402
from image_io import read_image  # noqa: E402
from models import ScreenKind  # noqa: E402
from vision import VisionDetector  # noqa: E402


PRIVATE_SAMPLE_FILES = (
    "01_ready.jpeg",
    "02_bite.jpeg",
    "03_bite.jpeg",
    "04_timing_left.jpeg",
    "05_timing_right.jpeg",
    "06_timing_cool.jpeg",
    "07_timing_perfect.jpeg",
    "08_result.jpeg",
    "09_result_confirm.png",
    "10_timing_green_flash.png",
    "11_result_no_button.png",
    "12_result_collect.png",
)
PRIVATE_SAMPLES_AVAILABLE = all((PROJECT / "samples" / name).is_file() for name in PRIVATE_SAMPLE_FILES)


class VisionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = VisionDetector(load_config(PROJECT / "config.json").vision)

    @unittest.skipUnless(PRIVATE_SAMPLES_AVAILABLE, "私人畫面樣本未包含在公開儲存庫")
    def test_reference_states(self) -> None:
        expected = {
            "01_ready.jpeg": ScreenKind.READY,
            "02_bite.jpeg": ScreenKind.BITE,
            "03_bite.jpeg": ScreenKind.BITE,
            "04_timing_left.jpeg": ScreenKind.TIMING,
            "05_timing_right.jpeg": ScreenKind.TIMING,
            "06_timing_cool.jpeg": ScreenKind.TIMING,
            "07_timing_perfect.jpeg": ScreenKind.TIMING,
            "08_result.jpeg": ScreenKind.RESULT,
            "09_result_confirm.png": ScreenKind.RESULT_CONFIRM,
            "10_timing_green_flash.png": ScreenKind.UNKNOWN,
            "11_result_no_button.png": ScreenKind.RESULT,
            "12_result_collect.png": ScreenKind.RESULT_CONFIRM,
        }
        for filename, kind in expected.items():
            with self.subTest(filename=filename):
                image = read_image(PROJECT / "samples" / filename, cv2.IMREAD_COLOR)
                self.assertIsNotNone(image)
                observation = self.detector.analyze(image)
                self.assertEqual(kind, observation.kind, observation.to_dict())

    @unittest.skipUnless(PRIVATE_SAMPLES_AVAILABLE, "私人畫面樣本未包含在公開儲存庫")
    def test_timing_samples_have_bar_measurements(self) -> None:
        expected_positions = {
            "04_timing_left.jpeg": (0.40, 0.45),
            "05_timing_right.jpeg": (0.72, 0.78),
            "06_timing_cool.jpeg": (0.66, 0.72),
            "07_timing_perfect.jpeg": (0.35, 0.42),
        }
        for filename, (left, right) in expected_positions.items():
            with self.subTest(filename=filename):
                image = read_image(PROJECT / "samples" / filename, cv2.IMREAD_COLOR)
                bar = self.detector.analyze(image).bar
                self.assertTrue(bar.valid, bar)
                self.assertGreaterEqual(bar.bobber_score, self.detector.config.min_bobber_score)
                self.assertGreaterEqual(bar.bobber_x, left)
                self.assertLessEqual(bar.bobber_x, right)

    @unittest.skipUnless(PRIVATE_SAMPLES_AVAILABLE, "私人畫面樣本未包含在公開儲存庫")
    def test_cool_and_stronger_feedback_have_bright_success_signal(self) -> None:
        regular = self.detector.analyze(read_image(PROJECT / "samples" / "04_timing_left.jpeg"))
        cool = self.detector.analyze(read_image(PROJECT / "samples" / "06_timing_cool.jpeg"))
        perfect = self.detector.analyze(read_image(PROJECT / "samples" / "07_timing_perfect.jpeg"))

        self.assertLess(regular.features["timing_success_white"], 0.17)
        self.assertGreaterEqual(cool.features["timing_success_white"], 0.17)
        self.assertGreater(perfect.features["timing_success_white"], cool.features["timing_success_white"])

    def test_non_game_colours_cannot_trigger_timing(self) -> None:
        frame = np.full((1536, 709, 3), (22, 22, 22), dtype=np.uint8)
        cv2.rectangle(frame, (145, 990), (560, 1080), (0, 230, 255), -1)
        cv2.rectangle(frame, (345, 970), (365, 1125), (0, 0, 235), -1)
        cv2.rectangle(frame, (205, 1110), (505, 1320), (210, 30, 190), -1)

        observation = self.detector.analyze(frame)

        self.assertEqual(ScreenKind.UNKNOWN, observation.kind, observation.to_dict())
        self.assertEqual(0.0, observation.features["active_game_scene"])

    def test_bobber_occlusion_does_not_split_yellow_target(self) -> None:
        frame = np.full((1536, 709, 3), (235, 190, 100), dtype=np.uint8)
        cv2.rectangle(frame, (220, 975), (540, 1100), (100, 55, 28), -1)
        cv2.rectangle(frame, (260, 1000), (500, 1070), (0, 235, 255), -1)
        cv2.rectangle(frame, (350, 980), (400, 1090), (0, 0, 235), -1)

        bar = self.detector.detect_bar(frame)

        self.assertTrue(bar.valid, bar)
        self.assertLess(bar.target_left, bar.bobber_x)
        self.assertGreater(bar.target_right, bar.bobber_x)

    def test_yellow_and_red_without_dark_track_is_not_a_timing_bar(self) -> None:
        frame = np.full((1536, 709, 3), (235, 190, 100), dtype=np.uint8)
        cv2.rectangle(frame, (260, 1000), (500, 1070), (0, 235, 255), -1)
        cv2.rectangle(frame, (365, 980), (389, 1090), (0, 0, 235), -1)

        bar = self.detector.detect_bar(frame)

        self.assertFalse(bar.valid, "沒有深藍軌道時，黃紅圖案不可被當成計時條")


if __name__ == "__main__":
    unittest.main()
