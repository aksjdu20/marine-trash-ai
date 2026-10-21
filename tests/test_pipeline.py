import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.common import NAMES, ROOT, label_path, read_labels, split_images, write_image
from src.metrics import match, scores
from src.prepare_dataset import prepare, split_groups
from src.underwater_augmentation import augment, build_training_copy
from src.validate_dataset import validate
from tests.make_fixture import make_fixture


class PipelineTest(unittest.TestCase):
    def test_bbox_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            f=Path(tmp)/"label.txt"
            for text in ["3 .5 .5 .2 .2", "0 .99 .5 .2 .2", "0 nan .5 .1 .1", "0 .5 .5 0 .2", "0 .5 .5 .2 .2 extra"]:
                f.write_text(text)
                with self.assertRaises(ValueError):
                    read_labels(f,3)
            f.write_text("0 .5 .5 .2 .2\n")
            self.assertEqual(read_labels(f,3).shape,(1,5))
            with self.assertRaises(ValueError):
                read_labels(f.parent/"missing.txt",3)

    def test_detection_errors_and_empty_predictions(self):
        gt=np.array([[0,0,10,10],[20,20,30,30],[40,40,50,50]])
        classes=np.array([0,1,0])
        pred=np.array([[0,0,10,10,.9,1],[20,20,30,30,.8,0],[80,80,90,90,.7,0]])
        cm,events=match(gt,classes,pred,3)
        self.assertEqual(cm.sum(),4)
        self.assertEqual({e["type"] for e in events},{"trash_as_marine","marine_as_trash","missed_trash","background_as_trash"})
        self.assertEqual(scores(cm,NAMES)[0]["recall"],0)
        cm,_=match(gt,classes,np.empty((0,6)),3)
        self.assertEqual(cm[:3,3].sum(),3)
        # 중복 탐지가 같은 GT를 두 번 맞춘 것으로 계산되면 실패
        pred=np.array([[0,0,10,10,.9,0],[0,0,10,10,.8,0]])
        cm,_=match(gt[:1],classes[:1],pred,3)
        self.assertEqual(cm[0,0],1)
        self.assertEqual(cm[3,0],1)
        # 오분류 박스가 같은 클래스의 TP 연결을 가로채면 안 됨
        pred=np.array([[0,0,10,10,.9,1],[0,0,9,10,.8,0]])
        cm,_=match(gt[:1],classes[:1],pred,3,class_aware=True)
        self.assertEqual(cm[0,0],1)
        self.assertEqual(cm[3,1],1)

    def test_group_leakage_blocked(self):
        rows=[{"group":"same","official":"train"},{"group":"same","official":"val"},
              {"group":"b","official":"train"},{"group":"c","official":"val"}]
        with self.assertRaises(ValueError):
            split_groups(rows,42,"official")
        split_groups(rows,42,"resplit")
        self.assertEqual(rows[0]["split"],rows[1]["split"])

    def test_augmentation_reproducible_and_boxes_preserved(self):
        im=np.full((32,32,3),100,dtype=np.uint8)
        a=augment(im,np.random.default_rng(42),.7,1)
        b=augment(im,np.random.default_rng(42),.7,1)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,im))
        np.testing.assert_array_equal(augment(im,np.random.default_rng(1),0,1),im)

    def test_full_prepare_and_validator(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            base=Path(tmp)
            raw=make_fixture(base/"raw")
            args=argparse.Namespace(output=str(base/"processed"),yolo=None,annotations=[str(raw/"instances_train.json"),str(raw/"instances_val.json")],images=str(raw/"images"),
                                    mapping=str(ROOT/"class_mapping.yaml"),inspect=False,two_classes=False,groups=None,group_regex=None,
                                    allow_image_split=False,split="official",seed=42,skip_review=False)
            out=prepare(args)
            d=out/"dataset.yaml"
            report=validate(d,base/"validation")
            self.assertTrue(report["valid"])
            self.assertEqual(report["images"],32)
            for split,n in [("train",24),("val",4),("test",4)]:
                self.assertEqual(len(split_images(d,split)),n)
            altered=build_training_copy(d,base/"aug",42,{"strength":.5,"probability":1})
            self.assertEqual(split_images(d,"val"),split_images(altered,"val"))
            self.assertEqual(split_images(d,"test"),split_images(altered,"test"))
            original=split_images(d,"train")[0]
            target=split_images(altered,"train")[0]
            self.assertEqual(label_path(original).read_bytes(),label_path(target).read_bytes())
            # 사용자 YOLO 입력 경로도 동일 원본/영상 그룹으로 재변환되어야 함
            with (out/"manifest.csv").open(encoding="utf-8-sig",newline="") as f:
                manifest=list(csv.DictReader(f))
            group_file=base/"groups.csv"
            with group_file.open("w",newline="",encoding="utf-8") as f:
                writer=csv.writer(f)
                writer.writerow(["file_name","video_id"])
                writer.writerows((Path(r["image"]).name,r["video_id"]) for r in manifest)
            args.yolo=str(d)
            args.annotations=None
            args.groups=str(group_file)
            args.output=str(base/"from_yolo")
            yolo_out=prepare(args)
            roundtrip=validate(yolo_out/"dataset.yaml",base/"yolo_check")
            self.assertEqual(report["fingerprint"],roundtrip["fingerprint"])
            # 분할 간 동일 이미지 복사 시 검증이 반드시 실패해야 함
            copy=out/"images/test/copy.png"
            copy.write_bytes(original.read_bytes())
            label_path(copy).write_bytes(label_path(original).read_bytes())
            with self.assertRaises(ValueError):
                validate(d,base/"bad_validation")

    def test_unknown_class_blocks_preparation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            base=Path(tmp)
            raw=make_fixture(base/"raw")
            filename=raw/"instances_train.json"
            data=json.loads(filename.read_text())
            data["categories"][0]["name"]="unidentified_object"
            filename.write_text(json.dumps(data))
            args=argparse.Namespace(output=str(base/"out"),yolo=None,annotations=[str(filename)],images=str(raw/"images"),mapping=str(ROOT/"class_mapping.yaml"),
                                    inspect=False,two_classes=False,groups=None,group_regex=None,allow_image_split=False,split="resplit",seed=42,skip_review=False)
            with self.assertRaises(ValueError):
                prepare(args)
            self.assertIn("검토/제외 클래스",(base/"out/review.csv").read_text(encoding="utf-8-sig"))

    def test_official_one_pixel_border_correction(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as tmp:
            base=Path(tmp)
            raw=make_fixture(base/"raw")
            filename=raw/"instances_train.json"
            d=json.loads(filename.read_text())
            d["annotations"][0]["bbox"]=[110,15,18.8,45]
            duplicate=dict(d["annotations"][0],id=999999)
            d["annotations"].append(duplicate)
            filename.write_text(json.dumps(d))
            args=argparse.Namespace(output=str(base/"out"),yolo=None,annotations=[str(filename),str(raw/"instances_val.json")],images=str(raw/"images"),
                                    mapping=str(ROOT/"class_mapping.yaml"),inspect=False,two_classes=False,groups=None,group_regex=None,
                                    allow_image_split=False,split="official",seed=42,skip_review=False,clip_border=True)
            out=prepare(args)
            with (out/"bbox_adjustments.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual(len(list(csv.DictReader(f))),2)
            record=json.loads((out/"preparation.json").read_text())
            self.assertEqual(record["duplicate_labels_removed"],1)
            with (out/"label_duplicates.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual(len(list(csv.DictReader(f))),1)
            d["annotations"][0]["bbox"]=[110,15,20,45]
            filename.write_text(json.dumps(d))
            args.output=str(base/"too_large")
            with self.assertRaises(ValueError):
                prepare(args)


if __name__=="__main__":
    unittest.main()
