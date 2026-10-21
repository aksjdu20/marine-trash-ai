import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Trash", "Marine_Life", "Equipment"]
COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]  # BGR
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def config(filename="config.yaml"):
    return yaml.safe_load(path(filename).read_text(encoding="utf-8"))


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config.yaml")
    return p


def save_json(filename, value):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def hash_file(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image(filename):
    # imread/imwrite가 일부 Windows 환경에서 한글 경로를 처리하지 못함
    image = cv2.imdecode(np.fromfile(str(filename), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"손상되었거나 읽을 수 없는 이미지: {filename}")
    return image


def write_image(filename, image):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix, image)
    if not ok:
        raise OSError(f"이미지 저장 실패: {p}")
    buf.tofile(str(p))


def pixel_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def read_labels(filename, nc):
    p = Path(filename)
    if not p.is_file():
        raise ValueError(f"라벨 파일 누락: {p}. 객체가 없는 이미지도 빈 txt가 필요합니다.")
    rows = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{p}:{line_no}: detection 형식 class x y w h 5개 필요 (polygon 미지원)")
        values = np.asarray([float(v) for v in fields])
        c, x, y, w, h = values
        if not np.isfinite(values).all() or c != int(c) or not 0 <= c < nc:
            raise ValueError(f"{p}:{line_no}: 잘못된 클래스/숫자")
        if w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
            raise ValueError(f"{p}:{line_no}: bbox가 이미지 범위를 벗어나거나 넓이가 0")
        rows.append(values.tolist())
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def xyxy(labels, width, height):
    boxes = labels[:, 1:5].copy()
    if len(boxes):
        boxes[:, :2] -= boxes[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, :2]
        boxes *= [width, height, width, height]
    return boxes


def label_path(image):
    parts = list(Path(image).parts)
    if "images" not in parts:
        raise ValueError(f"images 폴더 아래의 이미지가 필요합니다: {image}")
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def dataset(filename):
    p = path(filename)
    if not p.is_file():
        raise FileNotFoundError(f"데이터 설정 없음: {p}. prepare_dataset을 먼저 실행하세요.")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    base = Path(d.get("path", p.parent))
    if not base.is_absolute():
        base = (p.parent / base).resolve()
    names = d["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return d, base, names


def split_images(filename, split):
    d, base, _ = dataset(filename)
    if not d.get(split):
        raise ValueError(f"{split} 분할이 없습니다. 독립 테스트셋을 먼저 준비하세요.")
    entries = d[split] if isinstance(d[split], list) else [d[split]]
    images = []
    for entry in entries:
        p = Path(entry)
        p = p if p.is_absolute() else base / p
        if p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                q = Path(line.strip())
                if line.strip():
                    images.append(q.resolve() if q.is_absolute() else (p.parent / q).resolve())
        elif p.is_dir():
            images.extend(sorted(q.resolve() for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS))
        else:
            raise ValueError(f"이미지 폴더 또는 목록 파일이 필요합니다: {p}")
    if not images:
        raise ValueError(f"{split} 이미지가 0개입니다.")
    if len(images) != len(set(images)):
        raise ValueError(f"{split} 이미지 목록에 중복 경로가 있습니다.")
    return images


def device(requested="auto"):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def environment():
    packages = {}
    for name in ["torch", "torchvision", "ultralytics", "numpy", "opencv-python", "pandas", "matplotlib", "PyYAML", "streamlit", "albumentations"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def load_model(weights, expected=None):
    from ultralytics import YOLO
    p = path(weights)
    if not p.is_file():
        raise FileNotFoundError(f"학습 가중치 없음: {p}. A/B 학습 후 best.pt를 지정하세요.")
    model = YOLO(str(p))
    names = [model.names[i] for i in range(len(model.names))]
    if names not in [NAMES[:2], NAMES] or (expected is not None and names != expected):
        raise ValueError(f"학습 클래스가 일치하지 않습니다: {names}. COCO 사전학습 모델은 해양 3클래스 모델이 아닙니다.")
    return model


def detections(result):
    return result.boxes.data.detach().cpu().numpy()[:, :6]  # x1 y1 x2 y2 conf class


def draw(image, boxes, names, fps=None, conf=None):
    out = image.copy()
    counts = [0] * len(names)
    for x1, y1, x2, y2, score, c in boxes:
        c = int(c)
        counts[c] += 1
        color = COLORS[c]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{names[c]} {score:.2f}", (max(0, int(x1)), max(16, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    status = f"Trash: {counts[0]}  Marine Life: {counts[1]}"
    if fps is not None:
        status += f"  FPS: {fps:.1f}"
    if conf is not None:
        status += f"  conf: {conf:.2f}"
    cv2.putText(out, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    return out


def run(main):
    try:
        main()
    except (ValueError, FileNotFoundError, OSError, ImportError, KeyError, yaml.YAMLError) as e:
        raise SystemExit(f"오류: {e}") from None
