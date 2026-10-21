"""실제 바다 데이터가 없는 환경의 연결 검사. 도형 데이터로 연구 정확도를 주장하지 않음."""
import json
from pathlib import Path

import cv2
import numpy as np

from src.common import path, write_image


def make_fixture(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(123)
    categories=[{"id":1,"name":"trash_plastic"},{"id":2,"name":"animal_fish"},{"id":3,"name":"rov"}]
    for split,n in [("train",24),("val",8)]:
        images,anns=[],[]
        for i in range(n):
            image=rng.integers(10,45,(128,128,3),dtype=np.uint8)
            # 정확한 bbox를 가진 3종 도형. 3클래스 경로 검사용일 뿐 실제 의미 학습이 아님.
            shapes=[(10,15,20,45),(50,65,40,22),(90,15,23,30)]
            cv2.rectangle(image,(10,15),(29,59),(0,0,200),-1)
            cv2.ellipse(image,(70,76),(19,10),0,0,360,(0,200,0),-1)
            cv2.rectangle(image,(90,15),(112,44),(200,0,0),-1)
            filename=f"{split}_{i:03d}.png"
            write_image(output/"images"/filename,image)
            images.append({"id":i,"file_name":filename,"width":128,"height":128,"video_id":f"synthetic_{split}_{i:03d}"})
            for j,box in enumerate(shapes):
                anns.append({"id":i*3+j,"image_id":i,"category_id":j+1,"bbox":list(box),"area":box[2]*box[3],"iscrowd":0})
        (output/f"instances_{split}.json").write_text(json.dumps({"images":images,"annotations":anns,"categories":categories}),encoding="utf-8")
    writer=cv2.VideoWriter(str(output/"fixture.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),10,(128,128))
    if not writer.isOpened():
        raise OSError("합성 영상 writer 실패")
    for i in range(10):
        from src.common import read_image
        writer.write(read_image(output/"images"/f"train_{i:03d}.png"))
    writer.release()
    (output/"SYNTHETIC_ONLY.txt").write_text("코드 실행 확인용 합성 도형. 바다/쓰레기 학습 및 성능 평가용 데이터가 아닙니다.",encoding="utf-8")
    return output


if __name__=="__main__":
    print(make_fixture(path("tests/fixtures/raw")))
