"""학습 모델로 이미지/영상/카메라 탐지. 정답 없이 정확도를 생성하지 않음."""
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.common import (IMAGE_EXTS, config, detections, device, draw, environment,
                        hash_file, load_model, parser, path, read_image, run,
                        save_json, write_image)


def predict_image(model,filename,cfg,conf):
    im = read_image(filename)
    selected = device(cfg["device"])
    result = model.predict(im,conf=conf,imgsz=cfg["train"]["imgsz"],device=selected,
                           iou=cfg["evaluation"]["nms_iou"],max_det=cfg["evaluation"]["max_det"],verbose=False)[0]
    boxes = detections(result)
    return draw(im,boxes,[model.names[i] for i in range(len(model.names))]),boxes,result.speed


def analyze(source,weights,output,cfg,conf=None,display=False,max_frames=None,webcam=False):
    conf = cfg["evaluation"]["conf"] if conf is None else conf
    if not 0 <= conf <= 1:
        raise ValueError("conf는 0~1")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"결과 폴더 존재: {output}. 새로운 --output 사용")
    model = load_model(weights)
    names = [model.names[i] for i in range(len(model.names))]
    output.mkdir(parents=True,exist_ok=True)
    metadata = {**environment(),"source":str(source),"weights":str(path(weights)),"weights_sha256":hash_file(path(weights)),
                "conf":conf,"imgsz":cfg["train"]["imgsz"],"device":device(cfg["device"]),"accuracy_evaluated":False,
                "note":"라벨 없는 예측. 낮은 신뢰도/0 탐지는 실제 객체 부재의 증거가 아닙니다.","status":"started"}
    save_json(output/"prediction.json",metadata)
    pred_columns = ["frame","time_seconds","variant","conf_threshold","x1","y1","x2","y2","confidence","class"]
    if not webcam and Path(str(source)).suffix.lower() in IMAGE_EXTS:
        annotated,boxes,speed = predict_image(model,path(source),cfg,conf)
        write_image(output/"prediction.jpg",annotated)
        with (output/"predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(pred_columns)
            for row in boxes:
                w.writerow([0,0,"custom",conf,*row[:5],names[int(row[5])]])
        metadata.update(status="completed",speed_ms=speed,frames=1)
        save_json(output/"prediction.json",metadata)
        return output/"prediction.jpg"
    camera = int(source) if webcam else str(path(source))
    if max_frames is not None and max_frames < 1:
        raise ValueError("--max-frames는 1 이상")
    cap = cv2.VideoCapture(camera)
    writer = None
    active_variant = "custom"
    record_index,frame_index = 0,0
    paused = False
    recording = not webcam
    stream_start = time.perf_counter()
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    frame_times = []
    try:
        if not cap.isOpened():
            raise OSError(f"카메라/영상 열기 실패: {source}. 카메라 권한/인덱스/파일 코덱을 확인하세요.")
        with (output/"predictions.csv").open("w",newline="",encoding="utf-8-sig") as pf, (output/"frames.csv").open("w",newline="",encoding="utf-8-sig") as ff:
            pw,fw = csv.writer(pf),csv.writer(ff)
            pw.writerow(pred_columns)
            fw.writerow(["frame","time_seconds","variant","conf_threshold","trash_count","marine_life_count","equipment_count","inference_ms","pipeline_ms","pipeline_fps"])
            last_frame = None
            while True:
                start = time.perf_counter()
                if not paused:
                    ok,frame = cap.read()
                    if not ok:
                        if webcam and frame_index == 0:
                            raise OSError("카메라가 열렸지만 프레임을 읽지 못했습니다.")
                        break
                    if frame_index == 0:
                        model.predict(frame,conf=conf,imgsz=cfg["train"]["imgsz"],device=metadata["device"],verbose=False)
                        start = time.perf_counter()  # warmup 제외
                    result = model.predict(frame,conf=conf,imgsz=cfg["train"]["imgsz"],device=metadata["device"],
                                           iou=cfg["evaluation"]["nms_iou"],max_det=cfg["evaluation"]["max_det"],verbose=False)[0]
                    boxes = detections(result)
                    stamp = frame_index/source_fps if not webcam else time.perf_counter()-stream_start
                    counts = [int(np.sum(boxes[:,5]==c)) for c in range(3)]
                    # 최근 프레임 throughput로 화면 표시; 현재 프레임 측정은 아래에서 완료
                    fps = 1000/np.mean(frame_times[-30:]) if frame_times else 0
                    annotated = draw(frame,boxes,names,fps,conf)
                    cv2.putText(annotated,f"Inference: {result.speed['inference']:.1f} ms",(10,44),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    if writer is None and recording:
                        filename = output/"prediction.mp4" if record_index == 0 else output/f"recording_{record_index}.mp4"
                        writer = make_writer(filename,source_fps,frame)
                    if writer:
                        writer.write(annotated)
                    for row in boxes:
                        pw.writerow([frame_index,stamp,active_variant,conf,*row[:5],names[int(row[5])]])
                    if display:
                        cv2.imshow("Marine Trash AI: q quit | space pause | [ ] conf | s photo | r record | a/b model",annotated)
                    # capture/inference/draw/record/csv/display submission; waitKey 대기는 제외
                    elapsed = 1000*(time.perf_counter()-start)
                    frame_times.append(elapsed)
                    fw.writerow([frame_index,stamp,active_variant,conf,*counts,result.speed["inference"],elapsed,1000/max(elapsed,1e-9)])
                    last_frame = annotated
                    frame_index += 1
                    if max_frames is not None and frame_index >= max_frames:
                        break
                if display:
                    key = cv2.waitKey(30 if paused else 1)&0xff
                    if key in [ord("q"),27]:
                        break
                    if key == ord(" "):
                        paused = not paused
                    elif key == ord("["):
                        conf = max(0.01,round(conf-0.05,2))
                    elif key == ord("]"):
                        conf = min(0.99,round(conf+0.05,2))
                    elif key == ord("s") and last_frame is not None:
                        write_image(output/"snapshots"/f"frame_{frame_index:08d}_{time.time_ns()}.jpg",last_frame)
                    elif key == ord("r"):
                        recording = not recording
                        if writer:
                            writer.release()
                            writer = None
                        if recording:
                            record_index += 1
                    elif key in [ord("a"),ord("b")]:
                        variant = chr(key).upper()
                        try:
                            replacement = load_model(cfg["weights"][variant],names)
                            model = replacement
                            active_variant = variant
                            metadata.setdefault("model_switches",[]).append({"frame":frame_index,"variant":variant,"sha256":hash_file(path(cfg["weights"][variant]))})
                            if last_frame is not None:
                                model.predict(last_frame,imgsz=cfg["train"]["imgsz"],device=metadata["device"],verbose=False)
                        except (ValueError,FileNotFoundError) as e:
                            print(e)
            if frame_index == 0:
                raise OSError("분석 가능한 영상 프레임이 없습니다.")
        metadata.update(status="completed",frames=frame_index,pipeline_fps=1000/np.mean(frame_times),source_fps=source_fps,
                        timing_note="warmup and key wait excluded; includes capture/inference/draw/writer/CSV/display submission. OpenCV encoder/source FPS controls playback, not measured throughput.")
    except BaseException as e:
        metadata.update(status="interrupted" if isinstance(e,KeyboardInterrupt) else "failed",error=str(e),frames=frame_index)
        raise
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        save_json(output/"prediction.json",metadata)
    return output/"prediction.mp4" if not webcam else output


def make_writer(filename,fps,frame):
    h,w = frame.shape[:2]
    writer = cv2.VideoWriter(str(filename),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not writer.isOpened():
        raise OSError("mp4v 영상 저장 실패. OpenCV 코덱/경로를 확인하세요.")
    return writer


def main():
    p = parser(__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--variant",choices=["A","B"],default="A")
    p.add_argument("--weights")
    p.add_argument("--output",default="results/prediction")
    p.add_argument("--conf",type=float)
    p.add_argument("--display",action="store_true")
    p.add_argument("--max-frames",type=int)
    args = p.parse_args()
    cfg = config(args.config)
    print(analyze(args.source,args.weights or cfg["weights"][args.variant],path(args.output),cfg,args.conf,args.display,args.max_frames))


if __name__ == "__main__":
    run(main)
