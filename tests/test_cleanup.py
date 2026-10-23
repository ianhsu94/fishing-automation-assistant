from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from main import purge_debug_images  # noqa: E402


class CleanupTests(unittest.TestCase):
    def test_purge_permanently_deletes_all_debug_images_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            debug_dir = Path(temporary) / "debug"
            debug_dir.mkdir()
            for filename in ("first.png", "middle.jpg", "last.jpeg"):
                (debug_dir / filename).write_bytes(b"test image")
            marker = debug_dir / "keep.txt"
            marker.write_text("not a screenshot", encoding="utf-8")

            removed = purge_debug_images(debug_dir)

            self.assertEqual(3, removed)
            self.assertEqual([], list(debug_dir.glob("*.png")))
            self.assertEqual([], list(debug_dir.glob("*.jpg")))
            self.assertEqual([], list(debug_dir.glob("*.jpeg")))
            self.assertTrue(marker.exists(), "清理範圍只能包含截圖，不可誤刪其他檔案")


if __name__ == "__main__":
    unittest.main()
