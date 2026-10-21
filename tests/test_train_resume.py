import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common import NAMES, ROOT, config
from src.train import train


class ResumeTest(unittest.TestCase):
    def test_resume_preserves_conditions_and_augmented_data(self):
        for variant in ["A","B"]:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
                root=Path(tmp)
                cfg=config()
                pretrained=root/"models/pretrained"/cfg["model"]
                pretrained.parent.mkdir(parents=True)
                data=root/"dataset.yaml"
                data.write_text("test fixture")
                cfg["data"]=str(data)
                out=root/f"models/{variant}_seed42"
                def save_weights(**kwargs):
                    (out/"weights").mkdir(exist_ok=True)
                    (out/"weights/best.pt").write_bytes(b"best fixture")
                    (out/"weights/last.pt").write_bytes(b"last fixture")
                with patch("src.train.path",side_effect=lambda p:root/p), \
                     patch("src.train.validate",return_value={"fingerprint":"same data"}), \
                     patch("src.train.dataset",return_value=({},root,NAMES)), \
                     patch("src.train.device",return_value="cpu"), \
                     patch("src.train.environment",return_value={"utc":"test","packages":{"albumentations":None}}), \
                     patch("src.train.build_training_copy",return_value=data) as augment, \
                     patch("ultralytics.YOLO") as yolo:
                    yolo.return_value.task="detect"
                    yolo.return_value.train.side_effect=save_weights
                    def load_model(filename):
                        if not pretrained.exists():
                            pretrained.write_bytes(b"pretrained fixture")
                        return yolo.return_value
                    yolo.side_effect=load_model
                    train(cfg,variant,42)
                    record=json.loads((out/"experiment.json").read_text())
                    record["status"]="started"
                    (out/"experiment.json").write_text(json.dumps(record))
                    train(cfg,variant,42,resume=True)
                    self.assertEqual(yolo.call_args.args,(str(out/"weights/last.pt"),))
                    self.assertTrue(yolo.return_value.train.call_args.kwargs["resume"])
                    self.assertEqual(augment.call_count,1 if variant=="B" else 0)
                    resumed=json.loads((out/"experiment.json").read_text())
                    self.assertEqual(resumed["comparison_signature"],record["comparison_signature"])
                    self.assertEqual(len(resumed["resumptions"]),1)
                    yolo.reset_mock()
                    self.assertEqual(train(cfg,variant,42,resume=True),out)
                    yolo.assert_not_called()
                    changed=copy.deepcopy(cfg)
                    changed["train"]["batch"]=4
                    with self.assertRaisesRegex(ValueError,"원래 실험과 다릅니다"):
                        train(changed,variant,42,resume=True)
                    with self.assertRaisesRegex(ValueError,"학습 결과 존재"):
                        train(cfg,variant,42)


if __name__=="__main__":
    unittest.main()
