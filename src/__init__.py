"""해양 쓰레기 탐지 및 재현 가능한 실험 도구."""
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(_root / "results" / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(_root / "results" / ".matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
