"""평가 조건과 A/B 통제 조건을 검사한 뒤 실제 결과만 표/그래프로 비교."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import parser, path, run, save_json


def plot_training(training,output,label):
    if not training:
        return
    filename = Path(training)/"results.csv"
    if not filename.is_file():
        return
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    for suffix,cols,title in [("loss",[c for c in df if c.startswith("train/") and "loss" in c],"Training loss"),
                              ("validation",[c for c in df if c.startswith("metrics/")],"Validation metrics")]:
        if not cols:
            continue
        ax = df.plot(x="epoch",y=cols,title=f"{label} - {title}",figsize=(9,5))
        ax.figure.tight_layout()
        ax.figure.savefig(output/f"{label}_{suffix}.png",dpi=180)
        plt.close(ax.figure)


def compare(directories,output,allow_uncontrolled=False):
    if len(directories)<2:
        raise ValueError("평가 결과 폴더를 2개 이상 지정하세요.")
    metas,rows,classes,warnings = [],[],[],[]
    for directory in directories:
        directory = path(directory)
        meta = json.loads((directory/"evaluation.json").read_text(encoding="utf-8"))
        if meta.get("status") != "completed":
            raise ValueError(f"미완료 평가: {directory}")
        metas.append(meta)
        record = meta.get("training")
        if record is None:
            warnings.append(f"학습 조건 기록 없음: {directory}")
        label = directory.name
        summary = pd.read_csv(directory/"summary.csv").iloc[0].to_dict()
        rows.append({"run":label,"variant":record["variant"] if record else label,"seed":record["seed"] if record else None,**summary})
        per = pd.read_csv(directory/"per_class.csv")
        per["run"] = label
        per["variant"] = rows[-1]["variant"]
        classes.append(per)
    if len({m["evaluation_signature"] for m in metas}) != 1:
        raise ValueError("동일 테스트셋/장치/입력크기/라이브러리/평가 설정이 아닙니다. 비교 조건을 통일하세요.")
    if len({r["run"] for r in rows}) != len(rows):
        raise ValueError("결과 폴더의 마지막 이름이 중복됩니다. 고유한 run 폴더 이름을 사용하세요.")
    seed_groups = {}
    for m in metas:
        t = m.get("training")
        if t:
            seed_groups.setdefault(t["seed"],[]).append(t)
            if t.get("smoke") or t.get("fixture_only"):
                warnings.append("smoke/합성 검사 데이터: 연구 성능 비교로 해석하면 안 됩니다.")
    for seed,group in seed_groups.items():
        if len(group)<2 or {r["variant"] for r in group} != {"A","B"}:
            warnings.append(f"seed {seed}: A/B 쌍이 완전하지 않음")
        if len({r["comparison_signature"] for r in group})>1:
            warnings.append(f"seed {seed}: 학습 구조/초기 가중치/설정/데이터/장치가 다름")
    control_warnings = [w for w in warnings if "smoke/" not in w]
    if control_warnings and not allow_uncontrolled:
        raise ValueError("공정 비교 조건 미충족: "+"; ".join(control_warnings)+". 탐색용으로만 --allow-uncontrolled 사용 가능")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"비교 결과 폴더 존재: {output}")
    output.mkdir(parents=True,exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output/"comparison.csv",index=False,encoding="utf-8-sig")
    per = pd.concat(classes,ignore_index=True)
    per.to_csv(output/"class_comparison.csv",index=False,encoding="utf-8-sig")
    metric_cols = ["precision","recall","f1","map50","map50_95","inference_fps","pipeline_fps"]
    for metric in metric_cols:
        ax = df.plot.bar(x="run",y=metric,legend=False,rot=20,title=metric)
        if "fps" not in metric:
            ax.set_ylim(0,1)
        ax.figure.tight_layout()
        ax.figure.savefig(output/f"{metric}.png",dpi=180)
        plt.close(ax.figure)
    for metric in ["precision","recall","f1","map50","map50_95"]:
        ax = per.pivot(index="class",columns="run",values=metric).plot.bar(rot=0,title=f"Per-class {metric}")
        ax.set_ylim(0,1)
        ax.figure.tight_layout()
        ax.figure.savefig(output/f"class_{metric}.png",dpi=180)
        plt.close(ax.figure)
    stats = df.groupby("variant")[metric_cols].agg(["mean","std","count"])
    stats.to_csv(output/"seed_statistics.csv",encoding="utf-8-sig")
    paired = df.pivot(index="seed",columns="variant",values=metric_cols)
    if "A" in df["variant"].values and "B" in df["variant"].values:
        delta = pd.DataFrame({metric:paired[(metric,"B")]-paired[(metric,"A")] for metric in metric_cols})
        delta.to_csv(output/"paired_seed_delta_B_minus_A.csv",encoding="utf-8-sig")
    for row,meta in zip(rows,metas):
        plot_training(path(meta["weights"]).parent.parent,output,row["run"])
    save_json(output/"comparison.json",{"runs":[str(path(p)) for p in directories],"warnings":warnings,"controlled":not control_warnings,
                                        "interpretation":"B-A 차이와 여러 seed 평균/표준편차. 한 seed만으로 유의한 개선이라 단정하지 않기. 테스트셋으로 임계값/모델을 조정하지 않기."})
    print(df.to_string(index=False))
    return df


def main():
    p = parser(__doc__)
    p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--output",default="results/comparison")
    p.add_argument("--allow-uncontrolled",action="store_true")
    args = p.parse_args()
    compare(args.runs,path(args.output),args.allow_uncontrolled)


if __name__ == "__main__":
    run(main)
