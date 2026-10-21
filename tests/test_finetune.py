import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.common import ROOT, config
from src.run_finetune import finetune, report


class FinetuneTest(unittest.TestCase):
    def test_stages_resume_and_failure_are_recorded(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            root = Path(tmp)
            cfg = config("config.finetune.yaml")
            weights = root/"models/new/weights/best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"fixture")
            with patch("src.run_finetune.path", side_effect=lambda p:root/p), \
                 patch("src.run_finetune.train", return_value=weights.parent.parent) as tr, \
                 patch("src.run_finetune.evaluate") as ev, \
                 patch("src.run_finetune.report") as cp, \
                 patch("src.run_finetune.plot_training"):
                finetune(cfg, resume=True)
                self.assertTrue(tr.call_args.kwargs["resume"])
                self.assertEqual(tr.call_args.kwargs["run_name"], "B_finetune_seed42")
                ev.assert_called_once()
                cp.assert_called_once()
                status = json.loads((root/"results/finetune_status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["additional_labeled_camera_images"], 0)
            with patch("src.run_finetune.path", side_effect=lambda p:root/p), \
                 patch("src.run_finetune.train", side_effect=RuntimeError("training failed")):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    finetune(cfg)
                status = json.loads((root/"results/finetune_status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "failed")

    def test_report_uses_real_metrics_and_blocks_different_tests(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            root = Path(tmp)
            dirs = [root/"baseline", root/"current"]
            for i, directory in enumerate(dirs):
                directory.mkdir()
                (directory/"evaluation.json").write_text(json.dumps({"status":"completed", "evaluation_signature":"same"}))
                row = {m:.2+i*.1 for m in ["precision", "recall", "f1", "map50", "map50_95", "trash_detection_rate"]}
                pd.DataFrame([row]).to_csv(directory/"summary.csv", index=False)
                pd.DataFrame([{"class":"Trash", **row}]).to_csv(directory/"per_class.csv", index=False)
            report(*dirs, root/"comparison")
            record = json.loads((root/"comparison/comparison.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(record["delta_finetuned_minus_original"]["map50"], .1)
            self.assertFalse(record["controlled_ab_experiment"])
            (dirs[1]/"evaluation.json").write_text(json.dumps({"status":"completed", "evaluation_signature":"changed"}))
            with self.assertRaisesRegex(ValueError, "동일 test"):
                report(*dirs, root/"invalid")


if __name__ == "__main__":
    unittest.main()
