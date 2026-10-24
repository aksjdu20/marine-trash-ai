# 해양 쓰레기 식별 인공지능 개발 및 성능 평가

Python/PyTorch/Ultralytics YOLO 객체 탐지 연구 프로젝트입니다. 사진 속 개별 객체에 바운딩 박스·신뢰도·종류를 표시하고, 기본 모델 A와 수중 증강 모델 B를 독립 테스트셋에서 비교합니다. 로봇 제작은 포함하지 않습니다.

**2026-10-18: 공식 TrashCan 데이터로 A/B 각각 50 epoch 학습과 독립 테스트 평가·비교를 완료했습니다.** 실제 결과는 [한국어 결과 보고서](RESULTS_REPORT.md), [비교 CSV](results/comparison/comparison.csv), [비교 그래프](results/comparison/overview.png)에서 확인하세요. 학습된 가중치는 `models/A_seed42/weights/best.pt`, `models/B_seed42/weights/best.pt`이며 기본 설정에 연결돼 있습니다. B가 핵심 쓰레기·생물 구분을 전반적으로 개선했다는 결과는 나오지 않았습니다.

`models/pretrained/yolo26n.pt`는 일반 COCO 사전학습 모델이며 위 학습 가중치와 구분해야 합니다. `*_smoke`, `*_synthetic_*`, `tests/fixtures`, `results/smoke*`는 합성 도형으로 실행을 확인한 자료입니다. 그 수치/그래프는 실제 해양 탐지 성능으로 사용하지 마세요. 본 연구 수치는 실제 공식 데이터 학습·평가에서 생성했습니다.

### 2026-10-12 공식 데이터 학습

공식 University Digital Conservancy의 `dataset.zip`을 다운로드하고 Instance 버전 7,212장의 실제 클래스 목록을 확인했습니다. 원본과 출처/해시는 `data/raw/trashcan_1_0/`에 보관합니다. 실제 데이터 설정은 `config.yaml`에 연결되어 있습니다.

원본 공식 train/val에는 영상 ID 113개가 겹칩니다. 따라서 `vid_숫자_frame숫자.jpg`의 영상 ID로 70/15/15 근사 그룹 분할을 새로 만듭니다. 원본 박스 3,286개가 이미지 경계를 최대 0.9981픽셀 벗어나므로 `--clip-border`로 1픽셀 이내 초과만 잘라내고 `bbox_adjustments.csv`에 기록합니다. 원본 JSON은 변경하지 않습니다. 1픽셀보다 큰 초과는 검토 대상으로 차단합니다.

```powershell
python -m src.prepare_dataset --annotations data/raw/trashcan_1_0/dataset/instance_version/instances_train_trashcan.json data/raw/trashcan_1_0/dataset/instance_version/instances_val_trashcan.json --images data/raw/trashcan_1_0/dataset/instance_version --group-regex '^(vid_[0-9]+)_frame' --split resplit --clip-border --output data/processed/trashcan_instance_seed42
```

변환된 데이터는 `data/processed/trashcan_instance_seed42/dataset.yaml`입니다. **위 변환 명령은 이미 생성된 폴더에 다시 실행하지 않습니다.** 새로운 데이터 버전을 만들 때만 다른 output을 사용하세요.

```powershell
# A 50 epoch → 테스트 평가 → B 50 epoch → 테스트 평가 → A/B 비교
python -u -m src.run_research
# 현재 실행 상태와 로그
Get-Content results/research_status.json
Get-Content results/trashcan_training_stdout.log -Tail 10
Get-Content results/trashcan_training_stderr.log -Tail 10
```

실행 중인 학습을 중복 시작하지 마세요. Windows CPU에서는 수 시간 이상 걸릴 수 있으며, 컴퓨터를 끄거나 절전 상태로 두면 진행되지 않습니다. 완료 여부는 `research_status.json`의 `status: completed`와 각 모델의 `experiment.json`으로 확인하세요. 로그의 현재 epoch/시간만 보고 학습이 끝났다고 판단하지 않습니다. 원본 사용 시 저자 Hong/Fulton/Sattar와 JAMSTEC을 출처로 표기합니다. 자세한 진행 기록은 `OFFICIAL_TRAINING.md`를 확인하세요.

## 바로 시작하기 — 현재 작업 공간

이 작업 공간에서는 상위 폴더 `.venv`에 Python 실행 환경을 설치했습니다. VS Code에서 `C:\Coding Projects\현욱이 거`를 열고 터미널에 입력하세요.

```powershell
cd 'C:\Coding Projects\현욱이 거\marine_trash_ai'
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

브라우저에서 터미널에 표시되는 로컬 주소를 엽니다. 대시보드는 모델/데이터가 없어도 안내 화면을 표시합니다. 실제 이미지/영상 분석에는 학습한 `best.pt`가 필요합니다. `start_dashboard.ps1`도 실행할 수 있습니다.

이하 명령은 **marine_trash_ai 폴더에서 가상환경을 활성화한 상태**를 기준으로 합니다. 활성화하지 않아도 `python` 대신 가상환경 Python의 전체 경로를 쓰면 됩니다. 소스 파일을 `python src/train.py`로 실행하지 말고 **`python -m src.train`** 형식으로 실행하세요. 설정의 상대 경로는 현재 터미널 위치가 아니라 프로젝트 루트 기준으로 해석합니다.

## 프로젝트 구조

```text
marine_trash_ai/
  README.md, requirements.txt, requirements-lock.txt
  config.yaml                 모델/학습/증강/평가/가중치 설정
  class_mapping.yaml          원본 이름 → 통합 클래스 명시적 매핑
  experiments.yaml            clean/dark/turbid/mixed 영상 및 정답 경로
  app.py, start_dashboard.ps1  Streamlit 화면
  data/raw/                   다운로드 후 압축 해제한 원본 (수정하지 않음)
  data/processed/             변환 images, labels, dataset.yaml, manifest, analysis
  data/experiments/           수조 영상, 추출 프레임, 수동 정답
  models/                     사전학습 가중치, A/B seed별 학습 결과
  results/                    평가 CSV/그래프/오류 이미지/예측 영상/실험 기록
  src/
    common.py                 공통 경로, 입출력, 장치, 클래스 검사
    prepare_dataset.py        COCO/YOLO 변환 및 영상 그룹 분할
    validate_dataset.py       라벨/이미지/중복/누출 검증, 실제 분포 그래프
    train.py                  A/B 전이학습, 여러 seed, smoke
    underwater_augmentation.py OpenCV/NumPy 수중 광학 증강
    metrics.py, evaluate.py   객체 수준 오류, 고정 threshold P/R/F1, YOLO mAP
    compare_models.py         조건 검사, 표/그래프/seed 통계
    predict.py, webcam.py      사진/영상/웹캠, 임계값/사진/녹화
    experiments.py            실험 영상 프레임 추출, 조건별 평가
    annotate.py               마우스 드래그 정답 라벨 작성
  tests/                      데이터 없는 자동 검사와 선택적 YOLO 연결 검사
```

## 1. Python 설치와 가상환경

[Python 공식 다운로드](https://www.python.org/downloads/)에서 64비트 Python 3.10~3.12를 설치하세요. Windows 설치 시 PATH 옵션을 선택하고, VS Code Python 확장을 설치합니다. 가장 최신 Python 버전에서는 PyTorch wheel 지원이 늦을 수 있습니다.

새 컴퓨터의 Windows PowerShell:

```powershell
cd '프로젝트가 있는 경로\marine_trash_ai'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Mac/Linux:

```bash
cd marine_trash_ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

현재 만들어 둔 환경의 Windows 패키지 버전은 `requirements-lock.txt`에 저장했습니다. 같은 환경을 재현하려면 이 파일로 설치합니다. 다른 OS와 CUDA에서는 해당 플랫폼에 맞는 torch/torchvision을 먼저 설치하고 나머지는 `requirements.txt`로 설치하세요. 재현 기록에는 실제 사용한 `python -m pip freeze`도 보관합니다.

NVIDIA GPU를 사용하려면 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 운영체제/드라이버에 맞는 CUDA 설치 명령을 선택하세요. GPU가 있다고 CPU용 PyTorch가 자동으로 CUDA를 지원하는 것은 아닙니다. Apple Silicon은 MPS 지원 PyTorch가 필요합니다.

```bash
python -c "import torch; print('torch',torch.__version__,'CUDA',torch.cuda.is_available(),'MPS',torch.backends.mps.is_available())"
```

`device: auto`는 사용 가능한 CUDA → MPS → CPU 순서로 선택합니다. GPU가 없으면 CPU에서도 실행하지만 640 입력/50 epoch 학습은 오래 걸릴 수 있습니다. 먼저 smoke를 실행하세요. 다른 OS/CUDA/MPS의 실제 학습은 현재 컴퓨터에서 검증하지 못했습니다.

## 2. TrashCan 다운로드 및 실제 클래스 확인

[TrashCan 공식 안내](https://irvlab.cs.umn.edu/resources/trashcan)에서 연결된 [University of Minnesota 원본 저장소](https://conservancy.umn.edu/handle/11299/214865)의 파일을 직접 다운로드하고 `data/raw/`에 압축 해제하세요. 원본 배포물의 사용 조건을 확인합니다. **Material 또는 Instance 한 버전으로 시작하세요.** 두 버전은 원본 이미지를 공유하므로 두 배의 독립 데이터가 아닙니다. [원본 논문](https://arxiv.org/abs/2007.08097)은 COCO 변환과 원본 train/val 분할을 설명합니다.

압축 안의 실제 파일명/폴더는 배포물에 따라 다릅니다. 아래의 `instances_train.json`, `instances_val.json`, `images`는 예시이므로 실제 경로로 바꾸세요. COCO JSON에는 `images`, `annotations`, `categories`, 각 객체의 `bbox=[x,y,width,height]`가 있어야 합니다. 세그멘테이션 마스크가 있어도 이 프로젝트는 **bbox detection**을 학습합니다. bbox가 없는 마스크 전용 배포본은 COCO detection bbox를 먼저 만들어야 합니다.

```bash
python -m src.prepare_dataset --annotations data/raw/instance/instances_train.json data/raw/instance/instances_val.json --images data/raw/instance --inspect --output results/category_inspection
```

터미널과 `results/category_inspection/categories.csv`에서 **실제 클래스 이름/ID**를 확인하고 `class_mapping.yaml`을 검토합니다. COCO ID가 1부터 시작하거나 불연속이어도 이름을 기준으로 정확히 변환합니다. `trash_plastic`, `trash_unknown_instance`, `animal_fish`, `plant`, `rov` 같은 허용된 이름만 명시적으로 매핑하며 새로운 이름을 prefix만 보고 임의 분류하지 않습니다. 공백/하이픈은 underscore로 정규화됩니다.

통합 클래스:

| ID | 이름 | 박스 색 |
|---|---|---|
| 0 | Trash | 빨강 |
| 1 | Marine_Life | 초록 |
| 2 | Equipment | 파랑 |

플라스틱병/그물/물고기 등을 각각의 세부 종으로 예측하는 프로젝트가 아닙니다. 원본 객체를 위 세 범주로 합칩니다. `unknown`은 기본적으로 검토 대상입니다. 원본 `trash_unknown_instance`는 쓰레기 범주로 매핑되지만, 원본 논문의 catch-all에는 불확실한 객체가 포함될 수 있으므로 대표 샘플을 육안 확인하세요.

## 3. 영상 그룹과 train/val/test 분할

COCO 이미지에 `video_id`, `video`, `sequence_id`가 있으면 그대로 사용합니다. 없는 경우 원본 영상 ID를 조사해서 CSV를 만드세요. 아래 `file_name`은 COCO `images[].file_name` 또는 YOLO 원본 파일 이름과 정확히 일치해야 합니다.

```csv
file_name,video_id
video001_frame0001.jpg,video001
video001_frame0002.jpg,video001
video002_frame0001.jpg,video002
```

```bash
python -m src.prepare_dataset --annotations data/raw/instance/instances_train.json data/raw/instance/instances_val.json --images data/raw/instance --groups data/raw/groups.csv
```

파일명에 영상 ID가 확실히 포함되면 `--group-regex '^(.+)_frame[0-9]+'`처럼 **첫 번째 괄호로 영상 ID를 캡처**할 수 있습니다. 파일명 규칙을 확인하지 않고 임의의 regex를 쓰지 마세요. 기본 프로그램은 영상 ID가 없으면 변환을 중단합니다. 서로 관련 없는 독립 사진임을 확인한 경우에만 `--allow-image-split`을 사용합니다. 이 경우 영상 프레임 누출 방지는 보장할 수 없으며 해당 선택을 기록합니다.

기본 `--split official`은 COCO JSON의 `split` 또는 JSON 파일/부모 폴더의 `train`, `val`, `test` 토큰, YOLO YAML 분할을 검토합니다. 공식 train/val만 있으면 train을 그대로 두고 **원본 val 영상 그룹을 절반씩 val/test로 분리**합니다. 공식 val 수치와 본 프로젝트 test 수치는 분할이 다르므로 직접 비교하면 안 됩니다. 공식 분할에 동일 영상/중복 픽셀이 섞여 있으면 중단하고 재분할을 요구합니다.

70/15/15를 목표로 새로 분할하려면 `--split resplit`을 추가합니다. 영상 그룹 수 기준 근사 비율입니다. 영상마다 이미지 수가 달라 정확한 이미지 비율은 달라질 수 있습니다. 최소 세 개의 독립 영상 그룹이 필요하며 실제 비율은 `analysis/split_counts.csv`에 저장됩니다. 정확한 영상 분리가 이미지 비율보다 우선입니다.

```bash
python -m src.prepare_dataset --annotations data/raw/instance/instances_train.json data/raw/instance/instances_val.json --images data/raw/instance --groups data/raw/groups.csv --split resplit
```

Equipment 라벨이 부족한 경우 `--two-classes`를 추가합니다. 이 모드는 `Trash, Marine_Life`를 유지하고 **Equipment가 라벨된 이미지는 전체 제외**합니다. 장비 라벨만 지워 배경으로 취급하는 방식은 쓰레기/생물 학습에 혼선을 줄 수 있기 때문입니다. 많은 ROV 이미지가 빠질 수 있으므로 분포를 반드시 확인하세요. `config.yaml`의 `classes`도 두 이름으로 바꾸고 A/B 모두 같은 새 데이터로 다시 학습합니다.

변환 폴더에는 `dataset.yaml`, `manifest.csv`, `categories.csv`, `duplicates.csv`, `review.csv`, `preparation.json`, `images/{train,val,test}`, `labels/{train,val,test}`가 생깁니다. 원본 이미지와 라벨은 수정하지 않습니다. 기존 결과를 덮어쓰지 않으므로 다시 변환할 때는 `--output data/processed_v2`처럼 새 폴더를 사용하고 `config.yaml`의 `data`도 변경하세요. `--inspect`의 출력 폴더도 본 변환 폴더와 별도로 사용합니다.

## 4. 사용자 YOLO 데이터와 검증

Kaggle/Roboflow 등에서 **YOLO object detection** 형식으로 내보낸 `data.yaml`도 받습니다. `names`에는 원본 클래스가 있어야 합니다. polygon segmentation은 detection으로 다시 내보내세요.

```text
my_yolo/
  data.yaml
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
  test/images/*.jpg
  test/labels/*.txt
```

`data.yaml` 예시 (상대 `path`는 YAML 파일이 있는 폴더 기준):

```yaml
path: .
train: train/images
val: valid/images
test: test/images
names: [trash_plastic, animal_fish, rov]
```

```bash
python -m src.prepare_dataset --yolo data/raw/my_yolo/data.yaml --groups data/raw/groups.csv --output data/processed_yolo
```

Ultralytics용 YAML 내 임의 다운로드 코드는 실행하지 않습니다. 일부 외부 YAML의 `path`/`../`가 현재 압축 해제 위치와 다르면 직접 수정하세요. `images` 폴더와 대응 `labels` 폴더가 필요합니다. 라벨 한 줄은 `class x_center y_center width height`이며 좌표는 0~1 정규화 값입니다. 객체가 없는 이미지에는 **직접 검토한 빈 txt**가 필요합니다. 누락 파일을 자동으로 배경 라벨로 보완하지 않습니다.

```bash
python -m src.validate_dataset
# 또는
python -m src.validate_dataset --data data/processed_yolo/dataset.yaml --output results/dataset_yolo
```

손상 이미지, 실제 이미지 크기와 COCO 크기 불일치, bbox 범위/0 면적/NaN, class ID, 누락 txt, 중복 bbox, 동일 픽셀 중복, manifest의 영상 분할 누출을 검사합니다. SHA256은 디코딩한 픽셀 기준이므로 다른 이름/파일형식의 동일 이미지도 찾습니다. JPEG 재압축/크롭/거의 같은 프레임은 자동 판별하지 못합니다. 영상 ID와 샘플 육안 검토를 병행하세요. **이미지에 라벨되지 않은 실제 객체가 존재하는지, 라벨 의미가 맞는지는 자동 검사가 확인할 수 없습니다.**

검토 대상은 `review.csv`에 남기고 기본적으로 중단합니다. 매핑/라벨을 고친 뒤 새 output으로 실행하세요. `--skip-review`는 문제 이미지를 전체 제외하고 제외 수를 기록합니다. 상충하는 중복 라벨은 한 버전만 사용해 해결하세요. Material/Instance를 합쳐 annotation을 자동 병합하지 않습니다.

같은 이미지 안에서 매핑된 클래스와 bbox 좌표가 완전히 같은 detection 라벨은 한 개만 유지하며 `label_duplicates.csv`에 제거 수를 기록합니다. 원본 annotation은 보존합니다. 공식 Instance에서는 ROV 박스 한 개가 이 조건에 해당합니다.

`analysis/class_counts.csv`는 클래스가 포함된 이미지 수와 객체 수를 구분합니다. 같은 이미지에 여러 클래스가 있으면 클래스별 이미지 수 합은 전체 이미지 수보다 커질 수 있습니다. `analysis/samples/`에 정답 박스가 표시된 이미지가 저장됩니다.

## 5. A/B 학습 — 먼저 소규모 확인

학습 프로세스가 종료됐고 `last.pt`가 남아 있다면, 원래 config/데이터를 유지한 상태에서 다음 명령으로 남은 학습을 이어갈 수 있습니다. **현재 학습 프로세스가 살아 있으면 재개 명령을 중복 실행하지 마세요.** 기존 로그와 체크포인트를 먼저 복사해 보관하세요. 완료되지 않은 epoch의 배치는 다시 처리하며 총 목표 epoch를 늘리지 않습니다. 원래 데이터/학습 설정이 달라지면 재개를 차단하고, B의 기존 수중 증강 데이터를 그대로 사용합니다.

```bash
python -m src.run_research --resume
# 한 모델만 재개할 때:
python -m src.train --variant A --resume
```

재개 기록은 모델 폴더 `experiment.json`의 `resumptions`에 체크포인트 SHA256과 실행 환경으로 저장합니다. 완료된 학습과 조건이 일치하는 완료 평가 결과는 재사용합니다. 미완료 평가 폴더는 자동으로 덮어쓰지 않으므로 로그를 검토하고 보관한 뒤 복구하세요.

[Ultralytics YOLO26 공식 문서](https://docs.ultralytics.com/models/yolo26)는 `yolo26n.pt` 전이학습을 지원합니다. 첫 실행에 인터넷 연결이 필요하며 가중치는 `models/pretrained/`에 저장됩니다. 기본 epoch 50, imgsz 640, AdamW, batch 8, seed 42입니다. `config.yaml`에서 변경하세요. `model: yolov8n.pt`로 대체할 때 A와 B 둘 다 바꿔 동일 구조로 비교합니다.

```bash
python -m src.train --variant both --smoke
python -m src.train --variant A
python -m src.train --variant B
# A와 B 순차 실행
python -m src.train --variant both
```

smoke는 train 최대 8장/val 최대 8장, 2 epoch, imgsz 128, batch 2로 검사하며 테스트 이미지는 학습에 사용하지 않습니다. 일반 결과는 `models/A_seed42`, `models/B_seed42`, smoke는 `models/A_seed42_smoke`, `models/B_seed42_smoke`에 저장됩니다. 기존 run이 있으면 덮어쓰지 않습니다. 필요하면 `--run-name A_revision2`를 사용하고 `config.weights.A`도 새 `best.pt`로 지정하세요.

A/B 공통 증강은 HSV, 이동/확대, 좌우반전, mosaic 등 config의 동일 값입니다. 설치된 Ultralytics가 추가하는 기본 동작은 학습 폴더 `args.yaml`에 기록됩니다. [Ultralytics 학습 설정](https://docs.ultralytics.com/modes/train)을 참고하세요. **기본 모델 A도 증강을 사용합니다.** 이 프로젝트는 Albumentations를 필수 설치하지 않습니다. 별도로 설치되어 있으면 Ultralytics의 선택적 기본 Blur/MedianBlur/ToGray/CLAHE 등이 A/B 모두에 적용될 수 있으므로 두 학습 사이에 패키지 환경을 바꾸지 마세요. [선택적 Albumentations 통합](https://docs.ultralytics.com/integrations/albumentations)을 참고하세요.

B는 OpenCV/NumPy로 학습 이미지를 **1:1 광학 변환**하여 추가 밝기/대비/파랑 또는 초록 색조/3×3 블러/약한 노이즈를 적용합니다. bbox와 이미지 개수는 유지됩니다. 증강 복사본은 `results/augmented/B_seed42`에, 좌우 원본/증강 비교는 `previews/`에 저장됩니다. **val/test는 원본 목록을 그대로 참조합니다.** 설정 `underwater.strength`, `probability`로 강도를 조절합니다. 이 버전은 seed마다 한 번 고정된 광학 변환을 만듭니다. 매 epoch 새로운 수중 변환을 적용하는 방식이 아니며 실제 물리적 탁도/산란 모델도 아닙니다.

```bash
python -m src.underwater_augmentation --image data/raw/example.jpg
```

`models/.../experiment.json`에 원본 데이터 fingerprint, 초기/최종 가중치 해시, seed, train 설정, 장치, 라이브러리 버전, 증강 설정, 완료/실패 상태가 저장됩니다. `results.csv`, `results.png`, `args.yaml`, `weights/best.pt`, `weights/last.pt`도 확인하세요. 모델 선택은 **val 결과**로 하며 test를 보고 학습/threshold를 고르지 않습니다. `patience: 0`은 A/B의 실제 학습 epoch를 같게 유지합니다. 장치/라이브러리가 달라지면 seed만 같아도 완전한 bitwise 재현은 보장되지 않습니다.

## 6. 정답 테스트셋 평가와 A/B 비교

```bash
python -m src.evaluate --variant A
python -m src.evaluate --variant B
python -m src.compare_models --runs results/evaluation/A_seed42 results/evaluation/B_seed42
```

다른 가중치/데이터/출력도 직접 지정할 수 있습니다.

```bash
python -m src.evaluate --weights models/A_seed42/weights/best.pt --data data/processed/dataset.yaml --output results/evaluation/A_run2
```

지표의 의미:

- Precision: 탐지한 객체 중 클래스와 위치가 맞은 비율. TP/(TP+FP).
- Recall: 정답 객체 중 맞게 탐지한 비율. TP/(TP+FN).
- F1: Precision과 Recall의 조화평균. 2TP/(2TP+FP+FN).
- IoU: 예측/정답 박스 교집합 ÷ 합집합. 같은 클래스이며 IoU≥`match_iou`일 때 1:1 TP로 연결합니다.
- P/R/F1: config의 고정 conf와 match IoU, 전체 클래스 micro 값. 클래스별 값도 저장합니다. 정답이 없는 클래스의 recall/AP는 빈칸으로 남깁니다.
- mAP50 / mAP50-95: Ultralytics detection validator가 낮은 `ap_conf`로 PR 곡선을 만들고 IoU .50 및 .50:.95(간격 .05)의 AP를 계산합니다. 분류 accuracy와 다른 지표입니다.
- `ultralytics_precision_at_best_f1`, `ultralytics_recall_at_best_f1`: native PR 곡선의 best F1 지점 값으로 고정 conf 값과 구분해 저장합니다.
- Confusion Matrix/오류: 클래스 무관 IoU 연결로 **해양 생물→쓰레기** 등의 오분류를 드러냅니다. 행은 실제 클래스, 열은 예측 클래스, 마지막 Background 행/열은 FP/FN입니다. P/R 계산의 클래스 일치 연결과 방식이 달라 혼동행렬에서 P/R을 그대로 재계산하면 겹친 객체에서 다를 수 있습니다.
- `nms_iou`: 후처리 설정이며 정답 매칭 IoU와 다릅니다. YOLO26의 head/NMS 기본값은 설치된 Ultralytics 버전에 따르고 A/B 동일하게 사용합니다.

각 평가 폴더의 주요 파일:

| 파일 | 용도 |
|---|---|
| summary.csv | 전체 P/R/F1/mAP/FPS, 주요 오류 횟수 |
| per_class.csv | 클래스별 TP/FP/FN/support/P/R/F1/AP |
| confusion_matrix.csv/.png | 고정 threshold 혼동행렬 |
| errors.csv | missed_trash, marine_as_trash, trash_as_marine, background_as_trash 등 |
| error_images/ | 대표 오류 이미지, 정답 노랑/예측 클래스별 색 |
| predictions.csv, timing.csv | 예측 좌표/점수와 이미지별 처리 시간 |
| evaluation.json | 테스트 해시/가중치/평가 조건/학습 기록 |
| ultralytics/ | native PR 곡선/혼동행렬 등 |

`background_as_trash`는 GT와 매칭되지 않은 Trash 예측이며 진짜 배경, 장비, 중복 탐지, GT 누락 등을 포함할 수 있습니다. 원본 이미지와 라벨을 함께 검토하세요. `missed_trash`와 `trash_as_marine`를 따로 기록하므로 총 쓰레기 FN은 고정 클래스별 표도 확인합니다.

`inference_fps`는 워밍업을 제외한 순수 모델 추론(ms)의 역수입니다. `pipeline_fps`에는 이미지 읽기, 추론, 정답 매칭, 제한된 오류 이미지 저장 등이 들어갑니다. native 전체 val와 마지막 그래프/CSV 생성 시간은 제외합니다. FPS는 데이터 크기와 저장 장치/하드웨어에 영향을 받으므로 같은 조건에서 비교하세요. 초기 모델 다운로드/로드/워밍업은 FPS로 쓰지 않습니다.

비교 폴더에 P/R/F1/mAP/FPS, 클래스별 성능, epoch별 train loss/validation 그래프, `comparison.csv`, `seed_statistics.csv`, `paired_seed_delta_B_minus_A.csv`가 저장됩니다. 동일 테스트 해시/평가 설정/장치/라이브러리와 seed별 A/B 학습 서명을 검사합니다. 학습 기록이 없거나 조건이 다르면 기본 중단합니다. `--allow-uncontrolled`는 탐색 표만 만드는 옵션이며 공정한 연구 비교로 주장하지 마세요.

반복 실험 예시:

```bash
python -m src.train --variant both --seeds 42 123 2026
python -m src.evaluate --weights models/A_seed123/weights/best.pt --output results/evaluation/A_seed123
python -m src.evaluate --weights models/B_seed123/weights/best.pt --output results/evaluation/B_seed123
python -m src.evaluate --weights models/A_seed2026/weights/best.pt --output results/evaluation/A_seed2026
python -m src.evaluate --weights models/B_seed2026/weights/best.pt --output results/evaluation/B_seed2026
python -m src.compare_models --runs results/evaluation/A_seed42 results/evaluation/B_seed42 results/evaluation/A_seed123 results/evaluation/B_seed123 results/evaluation/A_seed2026 results/evaluation/B_seed2026 --output results/comparison_seeds
```

표의 seed는 가중치 옆 `experiment.json`에서 읽습니다. 데이터 분할 seed는 고정하고 학습 seed만 바꿔 A/B 쌍을 비교합니다. 평균/표준편차와 같은 seed의 B-A 차이를 확인합니다. 한 번의 작은 수치 증가를 유의한 개선이라고 단정하지 마세요. 표본 독립성/수조 영상 차이/클래스 불균형도 보고서에 기록합니다.

## 실제 수중 촬영을 위한 추가 미세조정

**2026-10-24 추가 50 epoch 학습·테스트 평가·비교를 완료했습니다.** Precision은 46.87%→48.94%로 상승했지만 Recall은 46.29%→44.04%, mAP50은 39.96%→39.40%, 쓰레기 Recall은 50.36%→47.72%로 하락했습니다. 전반적 개선은 확인되지 않아 기본 A/B 모델 선택을 유지합니다. 실제 결과와 한계는 [추가 학습 결과 보고서](FINETUNING_RESULTS.md), [전후 비교 CSV](results/finetune_comparison/comparison.csv)에서 확인하세요. 아래는 실행에 사용한 명령이며, 완료된 학습을 다시 시작할 필요는 없습니다.

`config.finetune.yaml`은 기존 B의 `best.pt`에서 **추가 50 epoch** 학습하는 별도 실험입니다. 공식 학습 이미지 4,918장 전체와 수중 증강을 사용하며 검증 1,159장·테스트 1,135장은 기존 영상 그룹 분할을 유지합니다. 새로 라벨링한 촬영 이미지는 아직 0장입니다. 기존 50 epoch A/B 실험과 미세조정 결과를 혼동하지 마세요.

```powershell
python -X utf8 -u -m src.run_finetune
# 프로세스가 종료된 것을 확인한 뒤 중단 지점에서 재개할 때만 실행:
python -X utf8 -u -m src.run_finetune --resume
```

기존 최적 B에서 새 AdamW 옵티마이저로 시작하며 학습률은 0.0001, mosaic는 0.5, 입력 640, batch 8, seed 42입니다. 기존 A/B 가중치를 덮어쓰지 않습니다. 학습은 `models/B_finetune_seed42/`, 상태는 `results/finetune_status.json`, 완료 후 실제 test 평가와 탐색적 전후 비교는 `results/evaluation/B_finetune_seed42/`, `results/finetune_comparison/`에 저장됩니다. `best.pt`는 검증 성능으로 선정합니다. 추가 학습량·초기 가중치 등이 달라 기존 A/B의 증강 효과 실험처럼 통제된 비교는 아닙니다.

현재 이 실행은 완료됐습니다. 최초 `results/finetune_stdout.log`, `results/finetune_stderr.log`와 중단·재개별 `results/finetune_restart_*` 로그는 로컬에 보존했습니다. 최종 성공 실행의 로그 접두사는 `results/finetune_restart_20261024T051858Z`이며 종료 코드는 0입니다. 완료 여부와 가중치·평가 서명 검증은 `results/finetune_status.json`, `results/finetune_verification.json`에 기록했습니다. 다른 학습을 진행할 경우 살아 있는 프로세스를 중복 실행하지 마세요.

**학습·평가 완료 후** 추가 모델로 웹캠을 실행하는 명령입니다. 기본 `config.yaml`의 기존 A/B 선택은 유지합니다.

```powershell
.\start_webcam.cmd --config config.finetune.yaml --variant B --camera 0 --conf 0.25
```

실제 촬영 환경에서의 개선을 검증하려면 해당 수중 영상에 정답 바운딩 박스를 지정한 독립 평가셋이 필요합니다. 모델 예측을 정답으로 간주해 재학습하지 않습니다. 여러 영상에서 수집한 실제 촬영 라벨을 학습/검증/테스트에 영상 단위로 분리해 추가하는 것이 후속 단계입니다.

## 7. 이미지·영상·웹캠 실행

```bash
python -m src.predict --source data/raw/example.jpg --variant A --output results/example_image
python -m src.predict --source data/experiments/clean.mp4 --variant B --output results/example_video
python -m src.predict --source data/experiments/clean.mp4 --variant A --conf 0.35 --display --output results/display_video
python -m src.webcam --camera 0 --variant A
```

웹캠 실행 시 장치가 열리고 탐지가 시작됩니다. GUI 창을 클릭한 상태에서 키를 누릅니다. Windows에서는 프로젝트 폴더의 PowerShell에서 다음 명령으로도 실행할 수 있습니다.

```powershell
.\start_webcam.cmd --camera 0 --variant A --conf 0.25
```

`.cmd` 실행 파일은 PowerShell 스크립트 실행 정책을 변경하지 않아도 사용할 수 있습니다. 기존 `.ps1`에서 PSSecurityException이 발생하면 위 명령을 사용하세요. 개선 모델로 시작하려면 `--variant B`를 지정합니다. 카메라가 열리지 않으면 Windows 설정 → 개인정보 및 보안 → 카메라에서 데스크톱 앱 접근을 허용하고, 다른 화상회의 앱을 종료하거나 `--camera 1`을 시도하세요. 화면이 느리면 CPU 추론 속도가 원인일 수 있습니다.

| 키 | 기능 |
|---|---|
| q / Esc | 종료 및 녹화 파일 마무리 |
| c | 카메라 장치 종료/다시 시작 (종료 시 녹화도 마무리) |
| Space | 화면 분석 일시 중지/재개 |
| [ / ] | conf 0.05 감소/증가 |
| s | 현재 탐지 화면 JPG 저장 |
| r | 녹화 시작/종료 (영상 파일 분석은 기본 자동 저장) |
| a / b | config의 학습 모델 A/B로 전환 |

화면에 클래스별 박스/신뢰도, Trash/Marine Life 현재 개수, 최근 프레임 처리 FPS, 순수 추론 ms와 현재 A/B 모델·LIVE/PAUSED/CAMERA OFF·REC 상태가 표시됩니다. FPS는 화면 출력/키 대기를 제외한 처리 속도입니다. Space는 마지막 화면을 유지하며 분석만 일시정지하고, c는 카메라 장치를 해제합니다. 정지 화면의 박스와 개수는 마지막 분석 결과입니다. 각 프레임의 개수/FPS/conf/모델은 `frames.csv`, 객체 좌표는 `predictions.csv`에 저장됩니다. 현재 화면 개수는 추적된 고유 개체 수가 아닙니다. 같은 물체는 여러 프레임에 반복됩니다. webcam은 r을 누르기 전에는 영상 녹화를 하지 않습니다. 녹화 재개는 LIVE 상태에서 r을 누르세요. 사진은 `results/webcam/<실행번호>/snapshots/`, 탐지 영상은 같은 실행 폴더의 `recording_*.mp4`에 저장됩니다. mp4v 녹화는 입력 영상 FPS(카메라 미제공 시 30) 기준 재생되므로 느린 웹캠 녹화는 실제 시간과 재생 시간이 다를 수 있습니다. 파일은 종료/녹화 중지 시 마무리됩니다.

카메라 없이 사진/영상으로 테스트할 수 있습니다. 파일 분석은 `--display` 없이 GUI도 필요하지 않습니다. `--max-frames 20`으로 처리 프레임 수를 제한할 수 있습니다. 탐지 0건이면 임계값/모델/영상 조건과 미탐지를 점검하며 객체가 없다고 확정하지 않습니다.

## 8. 수조 조건 실험과 정답 작성

`clean.mp4`, `dark.mp4`, `turbid.mp4`, `mixed.mp4`를 `data/experiments/`에 저장합니다. 수조의 물체/카메라 위치·촬영 시간·해상도·프레임 추출 간격을 통제하고 조명/탁도만 가능한 한 따로 바꾸세요. 생물 모형은 실제 해양 생물 이미지와 달라 데이터 분포 변화가 있으므로 수조 성능과 실제 바다 성능을 구분합니다.

먼저 프레임을 일정 간격으로 추출합니다. 예시의 30은 **30프레임마다**이며 30fps 영상일 때 약 1초입니다.

```bash
python -m src.experiments extract --video data/experiments/clean.mp4 --output data/experiments/clean_frames --every 30
python -m src.annotate --data data/experiments/clean_frames/dataset.yaml
```

annotate 화면에서 0(Trash)/1(Marine_Life)/2(Equipment)를 누르고 마우스로 모든 객체의 bbox를 드래그합니다. u는 마지막 박스 취소, s는 저장(직접 검토한 빈 배경도 s), n은 저장 후 다음 이미지, q는 종료입니다. `labels/test/frame_00000000.txt`처럼 이미지와 같은 stem의 YOLO 라벨이 생깁니다. 기존 txt가 있으면 수정할 수 있습니다. 별도의 YOLO 라벨링 도구를 써도 됩니다. **미검토 프레임에 빈 라벨을 일괄 생성하지 마세요.** 2클래스 모델이라면 프레임 `dataset.yaml`의 `names`를 두 클래스로 바꾸고 연구 기준에 맞춰 정답을 작성합니다.

다른 영상도 같은 간격으로 추출/라벨링한 뒤 `experiments.yaml`의 각 `ground_truth`를 해당 `.../dataset.yaml`로 지정합니다. `null`은 정답 없음입니다.

```bash
python -m src.experiments run --spec experiments.yaml
```

각 조건/A/B 탐지 영상과 CSV를 저장합니다. 정답이 있는 조건에만 같은 conf/IoU/imgsz 기준 P/R/mAP, 쓰레기 탐지율(Trash recall), 해양 생물→쓰레기 횟수, FPS를 계산합니다. 정답이 없는 조건은 accuracy 열이 빈칸이고 영상/예측/FPS만 남습니다. 정답 YAML은 그 영상에서 추출한 프레임과 연결해야 합니다. 프로그램이 GT가 다른 영상의 라벨인지 의미적으로 판별해 주지 못하므로 extraction.json의 영상 해시와 파일을 확인하세요.

`results/experiments/conditions.csv`와 조건별 그래프를 보고서에 사용합니다. 오류 횟수는 **라벨링된 프레임의 객체 횟수**로, 조건마다 프레임 수/정답 물체 수가 다르면 단순 횟수만 비교하지 말고 recall/precision/support도 같이 보세요. 탁도 실험을 합성 블러만으로 대체했다고 주장하지 않습니다.

## 9. 대시보드

```bash
python -m streamlit run app.py --server.address 127.0.0.1
```

다섯 메뉴에서 이미지 업로드 탐지, 영상 업로드 탐지/다운로드, A/B 평가 비교, 조건별 실험 결과, 데이터 분포를 확인합니다. 모델/평가/데이터 경로를 화면에서 지정할 수 있습니다. 학습과 정답 평가는 터미널에서 먼저 실행합니다. 로컬 웹캠은 OpenCV 프로그램으로 제공합니다. 영상은 mp4v 코덱이므로 브라우저에서 재생이 안 되면 다운로드해서 VLC 등으로 재생하세요.

## 10. 테스트와 개발 검증

가벼운 검사 (실제 바다 데이터, 가중치, 카메라 없이 실행):

```bash
python -m compileall -q src tests app.py
python -m unittest discover -s tests -v
```

검사는 bbox 입력, 오분류/미탐/중복 탐지 집계, 같은 영상의 분할 누출 차단, 고정 seed 증강, COCO 변환과 라벨 보존, 미정 클래스 중단, 대시보드 5개 메뉴를 포함합니다.

선택적 전체 연결 검사 (공식 가중치 최초 다운로드 및 실제 CPU 학습/평가가 발생):

```bash
python -m tests.smoke
```

32개의 도형 이미지를 만들고 COCO→YOLO 변환→A/B 2 epoch 학습→평가/비교→이미지/영상 분석→프레임 추출→라벨 없음 평가 차단→합성 좌표 GT 평가→라벨 있는/없는 조건 비교까지 확인합니다. 결과는 `results/smoke_runs/<고유번호>`에 저장하고 모델은 `models/*_synthetic_<고유번호>`에 둡니다. **도형의 클래스 이름은 코드 경로를 검사하기 위한 것으로 해양 식별 능력을 검증하지 않습니다.** 반복 실행은 새 폴더로 저장됩니다.

실제 실행 검증 기록은 `VERIFICATION.md`에 있습니다. Python 3.12/Windows/CPU에서 공식 TrashCan A/B 학습과 추가 미세조정·테스트 평가를 완료했습니다. 실제 웹캠 화면/키보드/녹화, 수조 촬영, CUDA/MPS는 별도 하드웨어 검증이 필요합니다.

## 문제 해결

| 증상 | 조치 |
|---|---|
| python이 Store로 연결/명령 없음 | Python 공식 설치 후 터미널 재시작. 현재 공간은 `..\.venv\Scripts\python.exe` 직접 사용 |
| Activate.ps1 실행 정책 오류 | 활성화 없이 `.venv\Scripts\python.exe -m ...` 사용. 시스템 정책 변경 불필요 |
| start_webcam.ps1 실행 정책 오류 | `.\start_webcam.cmd --camera 0 --variant A --conf 0.25` 사용. 가상환경 자동 선택, 시스템 정책 변경 불필요 |
| No module named src | marine_trash_ai에서 `python -m src.기능명` 실행 |
| 패키지 없음 | 실행 Python과 설치 Python을 일치시키고 `python -m pip install -r requirements.txt` |
| CUDA false | NVIDIA 드라이버와 CUDA 지원 PyTorch 확인. `device: cpu`로도 사용 가능 |
| CUDA out of memory | batch 8→4→2, 필요시 imgsz 축소. A/B 설정을 함께 바꾸고 새 run |
| MPS 연산 오류 | `device: cpu` 변경 또는 PyTorch 업데이트 후 A/B 모두 재검사 |
| yolo26n 다운로드/모델 오류 | 인터넷/프록시와 ultralytics 버전 확인. 공식 yolo26n.pt를 models/pretrained에 직접 배치. 대체 model은 yolov8n.pt |
| 영상 ID 없음 | groups.csv 또는 확인한 filename regex 준비. 독립 사진만 allow-image-split |
| review.csv가 생성되고 중단 | 실제 categories와 매핑/좌표/누락 파일을 수정. 새 output으로 재실행 |
| 출력 폴더 존재 | 새 output/run-name 사용. 이전 실험 자료를 임의 덮어쓰지 않음 |
| test split 없음 | 독립 테스트셋을 준비. val 평가를 test라고 보고하지 않음 |
| Missing label file | 모든 객체에 정답 작성. 실제 빈 배경도 검토 후 빈 txt 저장 |
| 카메라 안 열림 | OS 카메라 권한, 다른 앱 점유, camera 0/1/2 확인 |
| OpenCV GUI 없음 | opencv-python-headless 제거 후 opencv-python 설치. headless 환경은 파일 분석만 사용 |
| 영상 저장/재생 안 됨 | 입력 코덱/출력 경로 확인. mp4v를 지원하는 플레이어로 다운로드 재생 |
| 한글 경로 깨진 로그 | UTF-8 터미널 사용. Windows Ultralytics 로그가 한글을 생략할 수 있으나 실제 경로는 YAML/JSON 확인 |
| 비교 조건 불일치 | 같은 데이터/장치/설정/초기 가중치/환경으로 다시 평가/학습. 기록 없는 모델은 공정 비교 불가 |

탐구 보고서에는 데이터 출처/버전, 검토·제외 수, 원본 영상 분리 방식과 실제 분할 수, 클래스 통합 기준, A/B 공통 증강과 B 추가 증강, 고정 seed/epoch/imgsz/batch, 가중치 해시, 평가 conf/match IoU/AP cutoff, 하드웨어/FPS 측정 범위, 반복 실험 편차, 실패 사례와 미검증 범위를 포함하세요. 실제 성능이 나오기 전에는 빈 결과를 그대로 두고 필요한 작업을 설명합니다.
