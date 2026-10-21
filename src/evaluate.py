"""정답이 있는 독립 test 이미지 평가, mAP와 고정 threshold 진단을 별도 기록."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.common import (config, dataset, detections, device, draw, environment,
                        hash_file, label_path, load_model, parser, path,
                        pixel_hash, read_image, read_labels, run, save_json,
                        split_images, write_image, xyxy)
from src.metrics import match, scores


def evaluate(weights,data,output,cfg,split="test",external=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"평가 결과 존재: {output}. 새로운 --output 사용")
    _, _, names = dataset(data)
    images = split_images(data,split)
    eval_cfg = cfg["evaluation"]
    if not 0 <= eval_cfg["conf"] <= 1 or not 0 < eval_cfg["match_iou"] <= 1:
        raise ValueError("conf는 0~1, match_iou는 0 초과~1")
    label_digests, total_gt = [],0
    for p in images:
        im = read_image(p)
        labels = read_labels(label_path(p),len(names))
        total_gt += len(labels)
        label_digests.append((pixel_hash(im),hash_file(label_path(p))))
    if total_gt == 0:
        raise ValueError("평가셋 전체에 정답 객체가 0개입니다. mAP 평가 불가; 예측 기능을 사용하세요.")
    if len({r[0] for r in label_digests}) != len(images):
        raise ValueError("평가셋 중복 이미지. 반복 집계로 성능을 왜곡하지 않도록 제거하세요.")
    # 내부 평가에서는 전체 분할 누출 검증. 외부 실험셋은 독립 test만 있을 수 있음.
    if not external:
        from src.validate_dataset import validate
        validate(data,output/"dataset_validation")
    output.mkdir(parents=True,exist_ok=True)
    # Ultralytics는 test 전용 평가에서도 train/val 키를 요구함.
    # 사용자 YAML의 download 코드/외부 경로도 넘기지 않고 이미 검증한 목록만 전달.
    listing = output/"evaluation_images.txt"
    listing.write_text("\n".join(str(p) for p in images)+"\n",encoding="utf-8")
    native_data = output/"evaluation_only.yaml"
    native_data.write_text(yaml.safe_dump({"path":str(output.resolve()),"train":None,"val":None,
                                           split:str(listing.resolve()),"names":names},sort_keys=False),encoding="utf-8")
    model = load_model(weights,names)
    selected = device(cfg["device"])
    imgsz = cfg["train"]["imgsz"]
    ec = {"imgsz":imgsz,"device":selected,"iou":eval_cfg["nms_iou"],"max_det":eval_cfg["max_det"],"verbose":False}
    # 낮은 cutoff로 PR 곡선을 구성. 운영 cutoff와 분리.
    native = model.val(data=str(native_data),split=split,conf=eval_cfg["ap_conf"],batch=1,
                       workers=0,plots=True,project=str(output),name="ultralytics",exist_ok=True,**ec)
    nc = len(names)
    cm = np.zeros((nc+1,nc+1),dtype=int)
    metric_cm = np.zeros_like(cm)
    errors,pred_rows,timing = [],[],[]
    saved = 0
    # 초기 모델 로딩/워밍업은 FPS에서 제외
    model.predict(read_image(images[0]),conf=eval_cfg["conf"],**ec)
    for i,p in enumerate(images):
        start = time.perf_counter()
        im = read_image(p)
        gt = read_labels(label_path(p),nc)
        result = model.predict(im,conf=eval_cfg["conf"],**ec)[0]
        boxes = detections(result)
        matched,events = match(xyxy(gt,im.shape[1],im.shape[0]),gt[:,0],boxes,nc,eval_cfg["match_iou"])
        cm += matched
        # P/R에는 같은 클래스의 정답/예측만 TP로 연결 (객체 탐지 기준).
        pr_matched,_ = match(xyxy(gt,im.shape[1],im.shape[0]),gt[:,0],boxes,nc,eval_cfg["match_iou"],class_aware=True)
        metric_cm += pr_matched
        for row in boxes:
            pred_rows.append({"image":str(p),"x1":row[0],"y1":row[1],"x2":row[2],"y2":row[3],"confidence":row[4],"class":names[int(row[5])]})
        for event in events:
            errors.append({"image":str(p),**event})
        if events and saved < eval_cfg["max_error_images"]:
            out = draw(im,boxes,names)
            # GT 노란색, 예측 클래스별 색상. 파일+errors.csv를 함께 확인.
            import cv2
            for (x1,y1,x2,y2),c in zip(xyxy(gt,im.shape[1],im.shape[0]),gt[:,0]):
                cv2.rectangle(out,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),1)
                cv2.putText(out,f"GT {names[int(c)]}",(int(x1),max(12,int(y1)+14)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
            write_image(output/"error_images"/f"{i:06d}_{p.stem}.jpg",out)
            saved += 1
        timing.append({"image":str(p),"inference_ms":float(result.speed["inference"]),"pipeline_ms":1000*(time.perf_counter()-start),"detections":len(boxes)})
    per = scores(metric_cm,names)
    # AP가 없는 클래스의 native maps fallback을 보고서에 쓰지 않음
    native_indices = {int(c):i for i,c in enumerate(native.box.ap_class_index)}
    for c,row in enumerate(per):
        idx = native_indices.get(c)
        row.update(map50=float(native.box.ap50[idx]) if idx is not None else None,
                   map50_95=float(native.box.ap[idx]) if idx is not None else None)
    pd.DataFrame(per).to_csv(output/"per_class.csv",index=False,encoding="utf-8-sig")
    labels = names+["Background"]
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(output/"confusion_matrix.csv",encoding="utf-8-sig")
    fig,ax = plt.subplots(figsize=(6,5))
    ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(nc+1),labels,rotation=20)
    ax.set_yticks(range(nc+1),labels)
    ax.set(xlabel="Predicted",ylabel="Ground truth",title=f"Fixed conf={eval_cfg['conf']} / IoU={eval_cfg['match_iou']}")
    for y in range(nc+1):
        for x in range(nc+1):
            ax.text(x,y,str(cm[y,x]),ha="center",va="center",color="black")
    fig.tight_layout()
    fig.savefig(output/"confusion_matrix.png",dpi=180)
    plt.close(fig)
    pd.DataFrame(errors,columns=["image","type","true_class","predicted_class","gt_index","prediction_index","iou"]).to_csv(output/"errors.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(pred_rows,columns=["image","x1","y1","x2","y2","confidence","class"]).to_csv(output/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(timing).to_csv(output/"timing.csv",index=False,encoding="utf-8-sig")
    tp = sum(r["tp"] for r in per)
    fp = sum(r["fp"] for r in per)
    fn = sum(r["fn"] for r in per)
    precision,recall = tp/(tp+fp) if tp+fp else 0.0,tp/(tp+fn)
    event_counts = Counter(r["type"] for r in errors)
    summary = {"precision":precision,"recall":recall,"f1":2*tp/(2*tp+fp+fn),
               "map50":float(native.box.map50),"map50_95":float(native.box.map),
               "ultralytics_precision_at_best_f1":float(native.box.mp),"ultralytics_recall_at_best_f1":float(native.box.mr),
               "inference_fps":1000/np.mean([r["inference_ms"] for r in timing]),
               "pipeline_fps":1000/np.mean([r["pipeline_ms"] for r in timing]),
               "images":len(images),"gt_objects":total_gt,"trash_detection_rate":per[0]["recall"],
               **{k:event_counts[k] for k in ["missed_trash","marine_as_trash","trash_as_marine","background_as_trash"]}}
    # 영상 전체 unique 객체 수가 아니라 이미지/프레임 단위 객체 탐지율
    pd.DataFrame([summary]).to_csv(output/"summary.csv",index=False,encoding="utf-8-sig")
    train_record_path = path(weights).parent.parent/"experiment.json"
    training = json.loads(train_record_path.read_text(encoding="utf-8")) if train_record_path.is_file() else None
    signature_fields = {"eval":eval_cfg,"imgsz":imgsz,"names":names,"split":split,"device":selected,
                        "packages":environment()["packages"],"test_hash":hashlib.sha256(json.dumps(sorted(label_digests)).encode()).hexdigest()}
    meta = {**environment(),"weights":str(path(weights)),"weights_sha256":hash_file(path(weights)),"data":str(path(data)),
            **signature_fields,"evaluation_signature":hashlib.sha256(json.dumps(signature_fields,sort_keys=True).encode()).hexdigest(),
            "training":training,"status":"completed","metrics_note":"P/R/F1: micro, class-aware one-to-one IoU matching at fixed conf. Confusion/error analysis: class-agnostic IoU matching to reveal misclassifications. mAP: Ultralytics detection validator (PR integration over IoU .50:.95).",
            "timing_note":"warmup excluded; inference_fps uses Ultralytics inference only; pipeline includes disk/GT matching and bounded error image saves, excludes native val and final CSV/graphs",
            "external":external}
    save_json(output/"evaluation.json",meta)
    print(pd.DataFrame([summary]).to_string(index=False))
    return summary


def main():
    p = parser(__doc__)
    p.add_argument("--variant",choices=["A","B"],default="A")
    p.add_argument("--weights")
    p.add_argument("--data")
    p.add_argument("--split",choices=["test","val"],default="test")
    p.add_argument("--external",action="store_true",help="독립 수조 실험처럼 test만 있는 dataset")
    p.add_argument("--output")
    args = p.parse_args()
    cfg = config(args.config)
    evaluate(args.weights or cfg["weights"][args.variant],args.data or cfg["data"],
             path(args.output or f"results/evaluation/{args.variant}_seed{cfg['seed']}"),cfg,args.split,args.external)


if __name__ == "__main__":
    run(main)
