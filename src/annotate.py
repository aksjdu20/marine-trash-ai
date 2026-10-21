"""독립 실험 프레임에 YOLO 정답 작성: 마우스 드래그, 0/1/2 클래스, u 취소, s 저장, n 다음, q 종료."""
import csv
from pathlib import Path

import cv2
import numpy as np

from src.common import COLORS, dataset, draw, label_path, parser, path, read_image, read_labels, run, split_images, xyxy


def annotate(data,split):
    _,base,names=dataset(data)
    files=split_images(data,split)
    window="GT: drag box | 0 Trash 1 Marine 2 Equipment | u undo | s save (incl empty) | n next | q quit"
    cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
    audit_file=base/"annotation_review.csv"
    try:
        for p in files:
            original=read_image(p)
            h,w=original.shape[:2]
            # 화면 표시 크기를 제한하되 좌표는 원본 비율에 맞춰 변환
            scale=min(1,1100/w,700/h)
            image=cv2.resize(original,(round(w*scale),round(h*scale)))
            sh,sw=image.shape[:2]
            lab=label_path(p)
            rows=read_labels(lab,len(names)).tolist() if lab.exists() else []
            state={"start":None,"end":None,"class":0,"saved":lab.exists()}
            def mouse(event,x,y,flags,param):
                x,y=max(0,min(x,sw-1)),max(0,min(y,sh-1))
                if event==cv2.EVENT_LBUTTONDOWN:
                    state["start"]=(x,y)
                    state["end"]=(x,y)
                elif event==cv2.EVENT_MOUSEMOVE and state["start"] is not None:
                    state["end"]=(x,y)
                elif event==cv2.EVENT_LBUTTONUP and state["start"] is not None:
                    x1,x2=sorted([state["start"][0],x])
                    y1,y2=sorted([state["start"][1],y])
                    if x2-x1>=2 and y2-y1>=2:
                        rows.append([state["class"],(x1+x2)/(2*sw),(y1+y2)/(2*sh),(x2-x1)/sw,(y2-y1)/sh])
                        state["saved"]=False
                    state["start"]=None
            cv2.setMouseCallback(window,mouse)
            while True:
                labels=np.asarray(rows).reshape(-1,5)
                boxes=np.column_stack([xyxy(labels,sw,sh),np.ones(len(labels)),labels[:,0]])
                shown=draw(image,boxes,names)
                status=f"{p.name} | class: {names[state['class']]} | {'SAVED' if state['saved'] else 'UNSAVED'}"
                cv2.putText(shown,status,(10,44),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
                if state["start"] is not None:
                    cv2.rectangle(shown,state["start"],state["end"],COLORS[state["class"]],1)
                cv2.imshow(window,shown)
                key=cv2.waitKey(20)&0xff
                if key in [ord("q"),27]:
                    return
                if key in [ord("0"),ord("1"),ord("2")]:
                    c=key-ord("0")
                    if c<len(names):
                        state["class"]=c
                elif key==ord("u") and rows:
                    rows.pop()
                    state["saved"]=False
                elif key==ord("s"):
                    # s는 모든 객체/배경을 직접 검토했다는 명시적 확인
                    lab.parent.mkdir(parents=True,exist_ok=True)
                    lab.write_text("".join(" ".join(str(int(v)) if i==0 else f"{v:.8f}" for i,v in enumerate(row))+"\n" for row in rows),encoding="utf-8")
                    existed=audit_file.exists()
                    with audit_file.open("a",newline="",encoding="utf-8-sig") as f:
                        writer=csv.writer(f)
                        if not existed:
                            writer.writerow(["image","objects","reviewed"])
                        writer.writerow([str(p.relative_to(base)),len(rows),True])
                    state["saved"]=True
                elif key==ord("n"):
                    if state["saved"]:
                        break
                    print("s로 저장한 뒤 n을 누르세요. 객체가 없어도 직접 검토 후 s.")
    finally:
        cv2.destroyAllWindows()


def main():
    p=parser(__doc__)
    p.add_argument("--data",required=True)
    p.add_argument("--split",choices=["train","val","test"],default="test")
    args=p.parse_args()
    annotate(args.data,args.split)


if __name__=="__main__":
    run(main)
