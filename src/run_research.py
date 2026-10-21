"""A/B 학습 후 실제 테스트 평가/비교까지 순차 실행하고 상태를 기록."""
import os
import json
from datetime import datetime, timezone

from src.common import config, hash_file, parser, path, run, save_json
from src.train import train
from src.evaluate import evaluate
from src.compare_models import compare


def research(cfg, resume=False):
    status_file = path("results/research_status.json")
    status = {"pid":os.getpid(),"status":"running","dataset":cfg["data"],"model":cfg["model"],
              "epochs_per_model":cfg["train"]["epochs"],"seed":cfg["seed"],"completed_stages":[]}
    evaluations = []
    try:
        for variant in ["A","B"]:
            status.update(stage=f"train_{variant}",updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(status_file,status)
            folder = train(cfg,variant,cfg["seed"],resume=resume)
            status["completed_stages"].append(f"train_{variant}")
            status.update(stage=f"evaluate_{variant}",updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(status_file,status)
            output = path(f"results/evaluation/{variant}_seed{cfg['seed']}")
            if resume and output.exists():
                meta = json.loads((output/"evaluation.json").read_text(encoding="utf-8"))
                if (meta["status"] != "completed" or meta["weights_sha256"] != hash_file(folder/"weights/best.pt")
                        or meta["eval"] != cfg["evaluation"] or meta["imgsz"] != cfg["train"]["imgsz"]):
                    raise ValueError("기존 평가가 미완료이거나 모델/평가 설정이 다릅니다. 결과를 보존하고 검토하세요.")
            else:
                evaluate(str(folder/"weights/best.pt"),cfg["data"],output,cfg)
            evaluations.append(str(output))
            status["completed_stages"].append(f"evaluate_{variant}")
        status.update(stage="compare",updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(status_file,status)
        compare(evaluations,path("results/comparison"))
        status["completed_stages"].append("compare")
        status.update(status="completed",stage="done",updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(status_file,status)
    except BaseException as e:
        status.update(status="interrupted" if isinstance(e,KeyboardInterrupt) else "failed",
                      error=str(e),updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(status_file,status)
        raise


def main():
    p = parser(__doc__)
    p.add_argument("--resume",action="store_true",help="중단된 학습을 체크포인트에서 재개하고 완료된 단계를 재사용")
    args = p.parse_args()
    research(config(args.config),args.resume)


if __name__ == "__main__":
    run(main)
