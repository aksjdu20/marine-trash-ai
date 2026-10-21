import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

from src.common import ROOT, config, hash_file
from src.run_research import research


class ResearchRunnerTest(unittest.TestCase):
    def test_resume_reuses_only_matching_completed_evaluations(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            root=Path(tmp)
            cfg=config()
            for variant in ["A","B"]:
                weight=root/f"models/{variant}/weights/best.pt"
                weight.parent.mkdir(parents=True)
                weight.write_bytes(b"fixture")
                output=root/f"results/evaluation/{variant}_seed42"
                output.mkdir(parents=True)
                meta={"status":"completed","weights_sha256":hash_file(weight),"eval":cfg["evaluation"],"imgsz":cfg["train"]["imgsz"]}
                (output/"evaluation.json").write_text(json.dumps(meta))
            with patch("src.run_research.path",side_effect=lambda name:root/name), \
                 patch("src.run_research.train",side_effect=lambda c,v,s,**kw:root/f"models/{v}") as tr, \
                 patch("src.run_research.evaluate") as ev, patch("src.run_research.compare"):
                research(cfg,resume=True)
                ev.assert_not_called()
                self.assertTrue(all(call.kwargs["resume"] for call in tr.call_args_list))
                cfg["evaluation"]["conf"]=.4
                with self.assertRaisesRegex(ValueError,"평가 설정이 다릅니다"):
                    research(cfg,resume=True)

    def test_all_stages_and_failure_record(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            root=Path(tmp)
            with patch("src.run_research.path",side_effect=lambda name:root/name), \
                 patch("src.run_research.train",side_effect=[root/"A",root/"B"]) as tr, \
                 patch("src.run_research.evaluate") as ev, patch("src.run_research.compare") as cp:
                research(config())
                self.assertEqual(tr.call_count,2)
                self.assertEqual(ev.call_count,2)
                self.assertEqual(cp.call_count,1)
                status=json.loads((root/"results/research_status.json").read_text())
                self.assertEqual(status["status"],"completed")
                self.assertEqual(status["completed_stages"],["train_A","evaluate_A","train_B","evaluate_B","compare"])
            with patch("src.run_research.path",side_effect=lambda name:root/name), \
                 patch("src.run_research.train",side_effect=RuntimeError("training failed")):
                with self.assertRaises(RuntimeError):
                    research(config())
                status=json.loads((root/"results/research_status.json").read_text())
                self.assertEqual(status["status"],"failed")
                self.assertEqual(status["error"],"training failed")


if __name__=="__main__":
    unittest.main()
