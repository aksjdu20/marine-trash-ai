"""동일 설정 A/B 전이학습. --smoke는 실제 연구 결과와 별도 저장."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

from src.common import (config, dataset, device, environment, hash_file, parser,
                        path, run, save_json, split_images)
from src.underwater_augmentation import build_training_copy
from src.validate_dataset import validate


def train(cfg, variant, seed, data=None, smoke=False, run_name=None, resume=False):
    from ultralytics import YOLO
    data = path(data or cfg["data"])
    name = run_name or f"{variant}_seed{seed}" + ("_smoke" if smoke else "")
    out = path("models") / name
    resuming = resume and out.exists()
    previous = json.loads((out/"experiment.json").read_text(encoding="utf-8")) if resuming else None
    if out.exists() and not resuming:
        raise ValueError(f"학습 결과 존재: {out}. --run-name으로 새 이름 지정 (기존 결과 보호)")
    report = validate(data,path("results")/"validation"/name)
    _, _, names = dataset(data)
    settings = copy.deepcopy(cfg["train"])
    selected = device(cfg["device"])
    if selected == "cpu":
        print("CPU 학습: 실행 가능하지만 매우 느릴 수 있습니다. 먼저 --smoke를 사용하세요.")
    if settings["batch"] == -1 and selected in ["cpu","mps"]:
        settings["batch"] = 4
    used = data
    if smoke and not resuming:
        smoke_dir = path("results")/"smoke_data"/name
        smoke_dir.mkdir(parents=True,exist_ok=False)
        d = {"path":str(smoke_dir),"names":names}
        for split in ["train","val"]:
            listing = smoke_dir/f"{split}.txt"
            listing.write_text("\n".join(str(p) for p in split_images(data,split)[:8])+"\n",encoding="utf-8")
            d[split] = str(listing)
        used = smoke_dir/"dataset.yaml"
        used.write_text(yaml.safe_dump(d,sort_keys=False),encoding="utf-8")
    if smoke:
        settings.update(epochs=2,imgsz=128,batch=2,workers=0,close_mosaic=0)
    if variant == "B" and not resuming:
        used = build_training_copy(used,path("results")/"augmented"/name,seed,cfg["underwater"])
    source = cfg["model"]
    if Path(source).suffix != ".pt":
        raise ValueError("전이학습은 사전학습 .pt 가중치가 필요합니다 (예: yolo26n.pt).")
    local = path(source)
    # weights 다운로드를 프로젝트 models/pretrained 아래에 관리
    pretrained = local if local.exists() else path("models/pretrained")/Path(source).name
    pretrained.parent.mkdir(parents=True,exist_ok=True)
    model = YOLO(str(pretrained)) if not resuming else None
    controlled = {"model":source,"pretrained_sha256":hash_file(pretrained),"train":settings,"seed":seed,
                  "device":selected,"data_fingerprint":report["fingerprint"],"names":names,"smoke":smoke}
    signature = hashlib.sha256(json.dumps(controlled,sort_keys=True).encode()).hexdigest()
    if resuming:
        if previous["comparison_signature"] != signature or previous["augmentation"] != (cfg["underwater"] if variant == "B" else None):
            raise ValueError("재개할 데이터/학습 설정이 원래 실험과 다릅니다. 원래 config와 데이터를 사용하세요.")
        if previous["status"] == "completed":
            if hash_file(out/"weights/best.pt") != previous["best_sha256"]:
                raise ValueError("완료 모델의 가중치 해시가 다릅니다.")
            return out
        used = path(previous["training_dataset"])
        if not used.is_file():
            raise ValueError("기존 학습 데이터 YAML 없음. 증강 데이터를 새로 생성하지 않고 중단합니다.")
    checkpoint = out/"weights/last.pt" if resuming else pretrained
    if resuming and not checkpoint.is_file():
        raise ValueError("재개할 last.pt 없음. 기존 실험을 덮어쓰지 않습니다.")
    model = YOLO(str(checkpoint)) if resuming else model
    if model.task != "detect":
        raise ValueError("객체 탐지 detect 모델을 지정하세요.")
    out.mkdir(parents=True,exist_ok=resuming)
    record = {**environment(),**controlled,"variant":variant,"comparison_signature":signature,
              "source_dataset":str(data),"training_dataset":str(used),"augmentation":cfg["underwater"] if variant == "B" else None,
              "common_augmentation":"config train augmentation + installed Ultralytics defaults; args.yaml contains resolved args",
              "status":"started","fixture_only": "tests" in data.parts}
    record["optional_albumentations_installed"] = record["packages"]["albumentations"] is not None
    if resuming:
        record = previous
        record.setdefault("resumptions",[]).append({**environment(),"checkpoint_sha256":hash_file(checkpoint),
                                                   "previous_status":previous["status"],"previous_error":previous.get("error")})
        record.update(status="started")
        record.pop("error",None)
    save_json(out/"experiment.json",record)
    try:
        model.train(data=str(used),project=str(path("models")),name=name,exist_ok=True,
                    seed=seed,device=selected,pretrained=True,plots=True,resume=resuming,**settings)
        record["status"] = "completed"
        record["best_sha256"] = hash_file(out/"weights/best.pt")
        save_json(out/"experiment.json",record)
        print(f"학습 완료: {out/'weights/best.pt'}")
    except Exception as e:
        record.update(status="failed",error=str(e))
        save_json(out/"experiment.json",record)
        raise
    return out


def main():
    p = parser(__doc__)
    p.add_argument("--variant",choices=["A","B","both"],default="A")
    p.add_argument("--data")
    p.add_argument("--seeds",type=int,nargs="+")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--run-name")
    p.add_argument("--resume",action="store_true",help="동일 설정의 last.pt에서 남은 epoch 재개; 완료 모델은 검증 후 재사용")
    args = p.parse_args()
    cfg = config(args.config)
    seeds = args.seeds or [cfg["seed"]]
    variants = ["A","B"] if args.variant == "both" else [args.variant]
    if args.run_name and (len(seeds)>1 or len(variants)>1):
        raise ValueError("--run-name은 한 모델/한 seed에서만 사용")
    for seed in seeds:
        for variant in variants:
            train(cfg,variant,seed,args.data,args.smoke,args.run_name,args.resume)


if __name__ == "__main__":
    run(main)
