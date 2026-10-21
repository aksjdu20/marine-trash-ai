"""기존 B 가중치 미세조정 -> 같은 test 평가 -> 탐색적 전후 비교. 원본 A/B 보존."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import config, device, environment, hash_file, parser, path, run, save_json
from src.compare_models import plot_training
from src.evaluate import evaluate
from src.train import train


def report(reference, current, output):
    directories = [path(reference), path(current)]
    metadata = [json.loads((d/"evaluation.json").read_text(encoding="utf-8")) for d in directories]
    if any(m["status"] != "completed" for m in metadata) or len({m["evaluation_signature"] for m in metadata}) != 1:
        raise ValueError("완료된 동일 test/평가 조건의 결과만 비교할 수 있습니다.")
    output = path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, per_class = [], []
    for label, directory in zip(["Original_B", "Finetuned_B"], directories):
        rows.append({"run":label, **pd.read_csv(directory/"summary.csv").iloc[0].to_dict()})
        per_class.append(pd.read_csv(directory/"per_class.csv").assign(run=label))
    df = pd.DataFrame(rows).set_index("run")
    df.to_csv(output/"comparison.csv", encoding="utf-8-sig")
    pd.concat(per_class, ignore_index=True).to_csv(output/"class_comparison.csv", index=False, encoding="utf-8-sig")
    metrics = ["precision", "recall", "f1", "map50", "map50_95", "trash_detection_rate"]
    ax = df[metrics].T.plot.bar(ylim=(0, 1), rot=20, figsize=(10, 5), title="Exploratory fine-tuning comparison")
    ax.figure.tight_layout()
    ax.figure.savefig(output/"comparison.png", dpi=180)
    plt.close(ax.figure)
    save_json(output/"comparison.json", {
        "runs":[str(d) for d in directories], "controlled_ab_experiment":False,
        "delta_finetuned_minus_original":{m:float(df.loc["Finetuned_B",m]-df.loc["Original_B",m]) for m in metrics},
        "note":"기존 최적 B에서 추가 학습한 탐색 실험. 초기 가중치/누적 학습량/학습률/증강 설정이 다릅니다. 같은 test를 다시 평가했으며 단일 seed의 차이로 유의성을 주장할 수 없습니다. 웹캠 GT 정확도는 측정하지 않았습니다."})
    print(df.to_string())


def finetune(cfg, resume=False):
    settings = cfg["finetune"]
    name = settings["run_name"]
    status_file = path("results/finetune_status.json")
    status = {**environment(), "pid":os.getpid(), "status":"running", "stage":"train",
              "run_name":name, "model":cfg["model"], "data":cfg["data"], "train":cfg["train"],
              "seed":cfg["seed"], "additional_labeled_camera_images":settings["additional_labeled_camera_images"]}
    save_json(status_file, status)
    try:
        folder = train(cfg, "B", cfg["seed"], run_name=name, resume=resume)
        status.update(stage="evaluate_test")
        save_json(status_file, status)
        output = path("results/evaluation")/name
        if resume and output.exists():
            meta = json.loads((output/"evaluation.json").read_text(encoding="utf-8"))
            if (meta["status"] != "completed" or meta["weights_sha256"] != hash_file(folder/"weights/best.pt")
                    or meta["eval"] != cfg["evaluation"] or meta["imgsz"] != cfg["train"]["imgsz"]
                    or meta["data"] != str(path(cfg["data"])) or meta["split"] != "test"
                    or meta["device"] != device(cfg["device"]) or meta["packages"] != environment()["packages"]):
                raise ValueError("기존 평가의 조건이 다릅니다. 미완료 결과를 별도로 보존한 후 재평가하세요.")
        else:
            evaluate(str(folder/"weights/best.pt"), cfg["data"], output, cfg)
        status.update(stage="compare")
        save_json(status_file, status)
        comparison = path("results/finetune_comparison")
        report(settings["reference_evaluation"], output, comparison)
        plot_training(folder, comparison, name)
        status.update(status="completed", stage="done", weights=str(folder/"weights/best.pt"),
                      weights_sha256=hash_file(folder/"weights/best.pt"), comparison=str(comparison))
        save_json(status_file, status)
    except BaseException as e:
        status.update(status="interrupted" if isinstance(e, KeyboardInterrupt) else "failed", error=str(e))
        save_json(status_file, status)
        raise


def main():
    p = parser(__doc__)
    p.set_defaults(config="config.finetune.yaml")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    finetune(config(args.config), args.resume)


if __name__ == "__main__":
    run(main)
