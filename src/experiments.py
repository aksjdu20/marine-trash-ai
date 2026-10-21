"""수조 영상 프레임 추출, 동일 프레임의 수동 GT 평가, 조건별 비교."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.common import NAMES, config, hash_file, parser, path, run, save_json, write_image
from src.evaluate import evaluate
from src.predict import analyze


def extract(video,output,every=30):
    if every<1:
        raise ValueError("--every는 1 이상")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"프레임 폴더 존재: {output}")
    output.mkdir(parents=True,exist_ok=True)
    cap = cv2.VideoCapture(str(path(video)))
    rows=[]
    try:
        if not cap.isOpened():
            raise OSError(f"영상 열기 실패: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:
            raise ValueError("영상 FPS 없음. 실제 프레임레이트를 확인하세요.")
        i=0
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            if i%every == 0:
                filename=f"frame_{i:08d}.png"
                write_image(output/"images"/"test"/filename,frame)
                rows.append({"image":f"images/test/{filename}","frame":i,"time_seconds":i/fps,"video_id":path(video).stem})
            i+=1
        if not rows:
            raise ValueError("영상 프레임 없음")
        (output/"labels/test").mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(output/"frames.csv",index=False,encoding="utf-8-sig")
        (output/"dataset.yaml").write_text(yaml.safe_dump({"path":str(output.resolve()),"test":"images/test","names":NAMES},sort_keys=False),encoding="utf-8")
        save_json(output/"extraction.json",{"video":str(path(video)),"video_sha256":hash_file(path(video)),"every":every,"source_fps":fps,
                                             "note":"labels/test/*.txt에 모든 프레임의 정답을 수동 작성. 미라벨을 빈 txt로 보완하지 않기. 2클래스 모델은 names도 2개로 변경."})
    finally:
        cap.release()
    print(f"추출 {len(rows)} 프레임: {output}. src.annotate로 모든 객체에 정답을 표시하세요.")


def run_conditions(filename,cfg):
    spec=yaml.safe_load(path(filename).read_text(encoding="utf-8"))
    conditions=spec["conditions"]
    if len({c["name"] for c in conditions})!=len(conditions):
        raise ValueError("조건 name은 고유해야 합니다")
    root=path(spec.get("output","results/experiments"))
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"실험 출력 존재: {root}. 새 output 지정")
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    class_names=None
    for condition in conditions:
        name=condition["name"]
        if not name.replace("_","").replace("-","").isalnum():
            raise ValueError("조건 name에는 문자/숫자/_/-만 사용")
        for variant in spec.get("variants",["A","B"]):
            out=root/name/variant
            weights=spec.get("weights",cfg["weights"])[variant]
            # 동영상에는 GT 없이 항상 탐지/예측만 저장
            analyze(condition["video"],weights,out/"prediction",cfg)
            row={"condition":name,"variant":variant,"labeled":bool(condition.get("ground_truth"))}
            prediction=json.loads((out/"prediction/prediction.json").read_text(encoding="utf-8"))
            row["video_pipeline_fps"]=prediction["pipeline_fps"]
            if condition.get("ground_truth"):
                from src.common import dataset
                _,_,names=dataset(condition["ground_truth"])
                if class_names is not None and names!=class_names:
                    raise ValueError("조건별 정답 클래스가 다릅니다")
                class_names=names
                summary=evaluate(weights,condition["ground_truth"],out/"evaluation",cfg,external=True)
                row.update(summary)
                row["ground_truth"]=str(path(condition["ground_truth"]))
            rows.append(row)
            pd.DataFrame(rows).to_csv(root/"conditions.csv",index=False,encoding="utf-8-sig")
    df=pd.DataFrame(rows)
    for metric in ["trash_detection_rate","marine_as_trash","precision","recall","map50","video_pipeline_fps"]:
        if metric not in df:
            continue
        values=df.dropna(subset=[metric])
        if values.empty:
            continue
        ax=values.pivot(index="condition",columns="variant",values=metric).plot.bar(rot=20,title=metric)
        if metric in ["trash_detection_rate","precision","recall","map50"]:
            ax.set_ylim(0,1)
        ax.figure.tight_layout()
        ax.figure.savefig(root/f"{metric}.png",dpi=180)
        plt.close(ax.figure)
    save_json(root/"experiment_conditions.json",{"specification":spec,"config":cfg,"status":"completed",
                                               "note":"정답이 없는 조건은 정확도 빈칸. 탐지 개수와 오분류 횟수는 프레임별 값으로 같은 물체가 반복될 수 있음. 조건간 GT 표본 수/간격을 통제."})
    print(df.to_string(index=False))


def main():
    p=parser(__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("extract")
    e.add_argument("--video",required=True)
    e.add_argument("--output",required=True)
    e.add_argument("--every",type=int,default=30)
    r=sub.add_parser("run")
    r.add_argument("--spec",default="experiments.yaml")
    args=p.parse_args()
    if args.command=="extract":
        extract(args.video,path(args.output),args.every)
    else:
        run_conditions(args.spec,config(args.config))


if __name__=="__main__":
    run(main)
