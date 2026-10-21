"""OpenCV 카메라: q 종료, c 장치 시작/종료, space 일시정지, [] 임계값, s 사진, r 녹화, a/b 모델."""
import time

from src.common import config, parser, path, run
from src.predict import analyze


def main():
    p = parser(__doc__)
    p.add_argument("--camera",type=int,default=0)
    p.add_argument("--variant",choices=["A","B"],default="A")
    p.add_argument("--weights")
    p.add_argument("--output")
    p.add_argument("--conf",type=float)
    args = p.parse_args()
    cfg = config(args.config)
    output = path(args.output or f"results/webcam/{time.time_ns()}")
    print(f"결과 저장: {output}\nq 종료 | c 카메라 시작/종료 | Space 일시정지 | [ ] 신뢰도 | s 사진 | r 녹화 | a/b 모델")
    analyze(args.camera,args.weights or cfg["weights"][args.variant],output,cfg,args.conf,
            display=True,webcam=True,variant="custom" if args.weights else args.variant)


if __name__ == "__main__":
    run(main)
