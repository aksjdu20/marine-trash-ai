"""선택적 실제 YOLO 연결 검사. 인터넷/CPU 시간 필요. 결과는 합성 도형 전용."""
import argparse
import json
import time
from pathlib import Path

import yaml

from src.common import ROOT, config, path, read_image, save_json
from src.compare_models import compare
from src.evaluate import evaluate
from src.experiments import extract, run_conditions
from src.predict import analyze
from src.prepare_dataset import prepare
from src.train import train
from tests.make_fixture import make_fixture


def main():
    tag=str(time.time_ns())
    out=path(f"results/smoke_runs/{tag}")
    raw=make_fixture(out/"raw")
    args=argparse.Namespace(output=str(out/"processed"),yolo=None,annotations=[str(raw/"instances_train.json"),str(raw/"instances_val.json")],images=str(raw/"images"),
                            mapping=str(ROOT/"class_mapping.yaml"),inspect=False,two_classes=False,groups=None,group_regex=None,
                            allow_image_split=False,split="official",seed=42,skip_review=False)
    data=prepare(args)/"dataset.yaml"
    cfg=config()
    cfg["train"]["imgsz"]=128
    dirs=[]
    for variant in ["A","B"]:
        model_dir=train(cfg,variant,42,str(data),smoke=True,run_name=f"{variant}_synthetic_{tag}")
        weight=model_dir/"weights/best.pt"
        cfg["weights"][variant]=str(weight)
        dirs.append(str(out/f"evaluation_{variant}"))
        evaluate(str(weight),str(data),out/f"evaluation_{variant}",cfg)
    compare(dirs,out/"comparison")
    image=analyze(str(raw/"images/train_000.png"),cfg["weights"]["A"],out/"image",cfg)
    video=analyze(str(raw/"fixture.mp4"),cfg["weights"]["A"],out/"video",cfg,max_frames=5)
    import cv2
    cap=cv2.VideoCapture(str(video))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==5
    cap.release()
    frames=out/"frames"
    extract(str(raw/"fixture.mp4"),frames,every=3)
    try:
        evaluate(cfg["weights"]["A"],str(frames/"dataset.yaml"),out/"unlabeled_evaluation",cfg,external=True)
        raise AssertionError("미라벨 평가를 허용하면 안 됨")
    except ValueError as e:
        assert "라벨 파일 누락" in str(e)
    # 합성 도형의 실제 생성 좌표에서만 GT 구성. 사용자 영상에는 이 동작을 하지 않음.
    label="0 0.15625 0.29296875 0.15625 0.3515625\n1 0.546875 0.59375 0.3125 0.171875\n2 0.79296875 0.234375 0.1796875 0.234375\n"
    for p in (frames/"images/test").glob("*.png"):
        (frames/"labels/test"/p.with_suffix(".txt").name).write_text(label,encoding="utf-8")
    spec={"output":str(out/"conditions"),"variants":["A","B"],"weights":cfg["weights"],
          "conditions":[{"name":"synthetic_labeled","video":str(raw/"fixture.mp4"),"ground_truth":str(frames/"dataset.yaml")},
                        {"name":"synthetic_unlabeled","video":str(raw/"fixture.mp4"),"ground_truth":None}]}
    spec_file=out/"conditions.yaml"
    spec_file.write_text(yaml.safe_dump(spec),encoding="utf-8")
    run_conditions(str(spec_file),cfg)
    df=__import__("pandas").read_csv(out/"conditions/conditions.csv")
    assert df.loc[~df["labeled"],"precision"].isna().all()
    save_json(out/"SMOKE_ONLY.json",{"passed":True,"synthetic_only":True,"note":"연구 성능 아님. 실제 해양 데이터/카메라/CUDA/MPS 검증 아님.","image":str(image),"video":str(video)})
    print(f"합성 데이터 전체 연결 검사 통과: {out}")


if __name__=="__main__":
    main()
