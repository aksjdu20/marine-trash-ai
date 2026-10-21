# 실행 검증 기록 — 2026-10-09 (한국 시간)

## 2026-10-18 공식 데이터 최종 확인

공식 TrashCan Instance 7,212장을 영상 그룹으로 분리해 A/B 각각 50 epoch를 실제로 학습하고, 동일한 독립 test 1,135장/정답 2,089개를 평가했습니다. 두 모델의 completed 상태, 1~50 epoch CSV, best.pt SHA256과 학습·평가 기록의 일치, 같은 학습/평가 조건 서명, 분할 간 영상 ID 교집합 0, 실제 평가 CSV/혼동행렬/오류 이미지/비교 그래프를 확인했습니다. 검증 기록은 `results/official_training_verification.json`, 실제 수치·한계는 `RESULTS_REPORT.md`입니다.

완료된 실제 산출물을 연결한 Streamlit 5개 메뉴와 전처리·재개·평가 매칭 검사 **11개가 통과**했습니다(`results/final_tests.log`). 자동 검사는 합성/mocked 입력으로 코드 경로를 확인하며, 실제 연구 성능의 근거는 별도로 완료된 공식 데이터 학습·테스트 파일입니다. 비교 요약 그래프도 시각적으로 확인했습니다. 데이터나 epoch를 줄이지 않았으며, 중단·재개 이력은 보존했습니다. 아래 10월 9일의 합성 검증 기록과 실제 연구 결과를 구분하세요.

## 확인된 환경

- Windows 11, Python 3.12.14, Intel Core Ultra 5 225H, CPU 실행.
- PyTorch 2.14.1+cpu / torchvision 0.29.1 / Ultralytics 8.4.174.
- OpenCV 5.0.0 / NumPy 2.5.3 / Pandas 3.0.6 / Matplotlib 3.11.2 / Streamlit 1.65.0.
- 실제 설치 목록은 `requirements-lock.txt`. CUDA false, 현재 Windows에서는 MPS 사용 불가.
- 일반 Python 설치가 없어 앱의 번들 Python을 기반으로 상위 `.venv`를 만들었음. venv의 ensurepip 실패를 설치 도구로 보완해 pip를 설치했고 패키지 의존성 검사도 수행.

## 단계별 확인

| 요청 단계 | 구현/확인 |
|---|---|
| 1 기본 환경 | 실제 Python 환경 생성, 의존성 설치, config/파일 구조 구성 |
| 2 전처리 | 합성 COCO 32장 → YOLO 변환, 영상 ID 기준 train 24/val 4/test 4, YOLO 입력 재변환 fingerprint 일치, 샘플/CSV/그래프 |
| 3 기본 학습 | 공식 `yolo26n.pt` 다운로드, 실제 PyTorch CPU A 모델 2 epoch |
| 4 사진/영상 | 실제 YOLO 추론 호출, JPG/MP4/CSV 생성, 출력 MP4 5프레임 재개방 검사 |
| 5 웹캠 | 코드 문법/공통 영상 경로 검사. 실제 카메라/GUI/키 입력은 미검증 |
| 6 수중 증강 | bbox 보존, 동일 seed 변환 재현, val/test 원본 유지, B CPU 2 epoch |
| 7 평가/비교 | native mAP/PR, 고정 conf P/R/F1, confusion/미탐 이미지/학습 곡선/비교 표 생성 |
| 8 조건별 실험 | 영상에서 4프레임 추출, 미라벨 평가 차단, 합성 GT 조건 A/B 평가, 라벨 없는 조건의 accuracy 열은 빈칸 |
| 9 UI | Streamlit AppTest로 5개 메뉴 실행 확인. 모델 없는 초기 화면 정상 |
| 10 전체 검사 | 자동 검사 7개 및 `tests.smoke` 전체 연결 검사 통과 |

Streamlit AppTest는 Windows asyncio의 로컬 소켓을 사용합니다. 제한된 샌드박스의 소켓 차단으로 검사 시작이 지연됐고, 원인을 traceback으로 확인한 뒤 로컬 소켓을 사용할 수 있는 실행 환경에서 7개 검사를 통과했습니다. 앱 코드 오류로 숨기지 않았습니다.

수조 실험의 test 전용 YAML은 Ultralytics가 `train`/`val` 키를 요구하는 호환성 문제가 있었습니다. 평가 코드는 검증한 이미지 목록으로 평가 전용 YAML을 만들며 `train: null`, `val: null`로 빈 학습 분할을 명시합니다. 이 YAML은 학습용이 아닙니다. 수정 후 전체 연결 검사를 다시 수행해 통과했습니다.

## 재실행 명령

현재 workspace에서:

```powershell
cd 'C:\Coding Projects\현욱이 거\marine_trash_ai'
..\.venv\Scripts\python.exe -m compileall -q src tests app.py
..\.venv\Scripts\python.exe -m pip check
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe -m tests.smoke
```

7개 검사는 bbox 유효성, 객체 매칭 오류/중복 예측, 그룹 누출, 증강 재현, 전체 변환/라벨 보존/중복 차단, 불명확한 클래스, 5개 대시보드 메뉴를 검사합니다. 전체 smoke는 가중치 다운로드와 학습 시간이 필요합니다.

## 실제로 통과한 전체 연결 검사 산출물

`results/smoke_runs/1791554667454794100/SMOKE_ONLY.json`의 `passed: true`로 확인합니다. 같은 폴더에 원본 도형/변환 데이터/이미지·영상 예측/평가/비교 그래프/조건별 결과가 저장됐습니다. 실행 로그는 `results/smoke_full_fixed.log`입니다. 처음 실패한 `results/smoke_full.log`와 `results/smoke_runs/1791554587012293300`도 실패 추적 자료로 보존했습니다. 완료 마커가 없는 실행을 성공 자료로 취급하지 않습니다.

가중치:

- `models/pretrained/yolo26n.pt`: 공식 일반 객체 사전학습 가중치.
- `models/A_seed42_smoke`, `models/B_seed42_smoke`: 첫 합성 실행 검사.
- `models/A_synthetic_1791554667454794100`, `models/B_synthetic_1791554667454794100`: 통과한 전체 연결 검사의 2 epoch 가중치.

**위 산출물은 전부 코드 경로 검사입니다. 실제 해양 쓰레기 인식 성능을 입증하지 않습니다.** 합성 검사에서 conf 0.25의 탐지가 0건이어도 실행 검사는 통과할 수 있습니다. 이 검사는 유의미한 인식 능력을 검사하는 것이 아니라 데이터→학습→평가→저장 연결, 정답 검증, 실제 라이브러리 호환성을 검사합니다. 임의의 연구 정확도는 생성하지 않았습니다.

## 2026-10-09 당시 후속 과제와 현재 상태

1. 공식 원본/COCO JSON 확보, 영상 ID/매핑 확인 및 자동 검증은 완료했습니다. 전체 라벨 누락 객체/의미의 육안 검토는 남아 있습니다.
2. A/B 50 epoch와 독립 test 비교는 완료했습니다. seed 42 한 쌍이며 여러 seed 반복 평가는 수행하지 않았습니다.
3. 실제 웹캠, 카메라 권한, GUI 키 입력/사진/녹화/모델 전환 검사.
4. clean/dark/turbid/mixed 수조 영상과 전체 평가 표본의 수동 GT.
5. 다른 OS/CUDA/MPS 환경 검증. 현재 CPU 성공이 GPU 동작을 입증하지는 않음.

Streamlit 메뉴는 AppTest로 확인했습니다. 실제 브라우저의 업로드 클릭/영상 재생은 수동 확인이 필요하며 mp4v 코덱은 일부 브라우저에서 지원되지 않을 수 있습니다. 영상 파일 자체는 OpenCV로 재개방해 확인했습니다.

## 2026-10-21 웹캠 조작 기능 보완

- c 키로 실제 카메라 장치를 해제/재개하며, 장치 종료 시 녹화를 마무리합니다. 화면에 현재 모델, 카메라/일시정지 상태, 녹화 상태와 조작 키를 표시합니다.
- `start_webcam.ps1`로 카메라 번호, A/B 모델, 초기 신뢰도를 지정할 수 있습니다.
- 장치/키 입력을 모의한 웹캠 테스트 2개를 포함해 다음 회귀 검사 12개가 통과했습니다. Windows 기본 cp949에서는 기존 테스트의 UTF-8 JSON 읽기에 오류가 있어 UTF-8 모드로 검사했습니다.

```powershell
python -X utf8 -m unittest tests.test_pipeline tests.test_research_runner tests.test_train_resume tests.test_webcam -v
```

- 실제 A 학습 가중치로 합성 영상 2프레임을 CPU 추론하고, 탐지 MP4를 저장한 뒤 OpenCV로 다시 읽었습니다. 이는 실행 경로 검사이며 인식 정확도를 평가한 실험이 아닙니다.
- 실제 카메라 연결, GUI 키 입력과 하드웨어 FPS는 확인하지 않았습니다. 이번 전체 검사 시 대시보드 AppTest가 종료되지 않아 중단했으며, 이번 변경의 회귀 통과 수에는 포함하지 않았습니다.
