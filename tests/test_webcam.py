"""카메라 없이 키 입력, 장치 수명, 저장 및 실패 시 정리를 검증합니다."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.common import ROOT, config
from src.predict import analyze


class WebcamTest(unittest.TestCase):
    def exercise(self, root, keys, opened=True, read_ok=True):
        cap = MagicMock()
        cap.isOpened.return_value = opened
        cap.get.return_value = 30
        cap.read.return_value = (read_ok, np.zeros((240, 640, 3), dtype=np.uint8))
        model = MagicMock()
        model.names = {0: "Trash", 1: "Marine_Life", 2: "Equipment"}
        model.predict.return_value = [MagicMock(speed={"inference": 2.5})]
        writer = MagicMock()
        with patch("src.predict.load_model", return_value=model), \
             patch("src.predict.environment", return_value={}), \
             patch("src.predict.hash_file", return_value="test"), \
             patch("src.predict.device", return_value="cpu"), \
             patch("src.predict.detections", return_value=np.array([[20, 80, 90, 150, .9, 0], [100, 80, 160, 150, .8, 1]])), \
             patch("src.predict.cv2.VideoCapture", return_value=cap), \
             patch("src.predict.make_writer", return_value=writer), \
             patch("src.predict.cv2.imshow") as show, \
             patch("src.predict.cv2.waitKey", side_effect=keys), \
             patch("src.predict.cv2.destroyAllWindows") as close:
            try:
                analyze(0, "test.pt", root, config(), display=True, webcam=True, variant="A")
            finally:
                cap.release.assert_called()
                close.assert_called_once()
        return cap, writer, show

    def test_controls_and_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            root = Path(tmp)/"output"
            cap, writer, show = self.exercise(root, [ord(x) for x in ["r", "s", "]", "b", " ", " ", "c", "c", "a", "r", "r", "q"]])
            cap.open.assert_called_once_with(0)
            self.assertGreaterEqual(cap.release.call_count, 2)
            writer.write.assert_called()
            writer.release.assert_called()
            self.assertEqual(len(list((root/"snapshots").glob("*.jpg"))), 1)
            with (root/"frames.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["variant"] for r in rows}, {"A", "B"})
            self.assertIn("0.3", {r["conf_threshold"] for r in rows})
            self.assertTrue(all(r["trash_count"] == r["marine_life_count"] == "1" for r in rows))
            record = json.loads((root/"prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "completed")
            self.assertEqual([s["variant"] for s in record["model_switches"]], ["B", "A"])
            self.assertGreater(show.call_count, len(rows))

    def test_camera_errors_are_reported_and_resources_closed(self):
        for opened, read_ok in [(False, True), (True, False)]:
            with self.subTest(opened=opened), tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
                root = Path(tmp)/"output"
                with self.assertRaises(OSError):
                    self.exercise(root, [], opened, read_ok)
                record = json.loads((root/"prediction.json").read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "failed")


if __name__ == "__main__":
    unittest.main()
