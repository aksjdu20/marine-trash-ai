# 공식 TrashCan 학습 기록 — 2026-10-12

**최종 완료: 2026-10-18 13:20 KST.** A/B 각각 50 epoch, 동일한 독립 test 1,135장 평가와 비교 그래프 생성을 모두 마쳤습니다. `results/research_status.json`은 completed입니다. 최종 결과와 해석은 [RESULTS_REPORT.md](RESULTS_REPORT.md), 파일·가중치 해시 검증은 `results/official_training_verification.json`, 자동 검사 11개 통과 기록은 `results/final_tests.log`에 있습니다. 완료 확인 자동화는 비활성화했습니다. 아래는 원본 출처와 실행·복구 이력을 그대로 보존한 기록입니다.

공식 안내: https://irvlab.cs.umn.edu/resources/trashcan

원본 저장소: https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7

DOI: https://doi.org/10.13020/g1gx-y834

실제 다운로드 파일: `data/raw/trashcan_1_0/dataset.zip`

SHA256: `caf93d992b85343cfc500386c94ca9ffa6c63e2284e04127753a40db8f33ba4c`

크기: 553,029,970 bytes. 원본 ZIP 및 Instance 이미지/JSON과 LICENSE를 보관했습니다. 자동 HTTP 요청은 Azure WAF 403을 받았지만, 일반 브라우저에서 정상 확인 후 공식 Download dataset.zip 링크로 다운로드했습니다. 비공식 재가공본은 사용하지 않았습니다.

원본 Instance에는 train 6,065장/val 1,147장과 22개 클래스, 객체 annotation 12,128개가 있습니다. `rov`는 Equipment, `plant`와 `animal_*`의 실제 7개 이름은 Marine_Life, 실제 14개 `trash_*` 이름은 Trash로 명시적으로 매핑했습니다. 기존 매핑에서 빠졌던 trash_clothing/trash_pipe/trash_snack_wrapper/trash_can은 실제 categories를 확인하고 추가했습니다. 의미 불명확한 새 클래스는 없습니다.

원본 파일명은 `vid_000159_frame0000008.jpg` 형태입니다. 원본 train에는 영상 310개, val에는 115개가 있고 영상 113개가 겹칩니다. 원본 split을 유지하면 근접 프레임의 학습/평가 누출 우려가 있어 `^(vid_[0-9]+)_frame`으로 영상 그룹을 추출하고 seed 42로 다시 분할했습니다. Material 버전을 함께 사용하지 않았습니다.

원본 bbox 3,286개는 이미지 경계를 1픽셀 미만 벗어납니다. 최대 초과는 0.9980392156862763px이며 1픽셀보다 큰 초과/0면적/음수 시작 좌표는 확인되지 않았습니다. 이 작은 경계 초과만 이미지와의 교집합으로 잘라내고 변환 폴더의 `bbox_adjustments.csv`에 원본/수정 좌표를 저장합니다. 원본 COCO JSON은 보존했습니다. 1픽셀 초과는 자동 수정하지 않으며 전처리 검토를 요구합니다.

변환/검증 결과: `data/processed/trashcan_instance_seed42/analysis/validation.json`, `split_counts.csv`, `class_counts.csv`. 이미지/영상 분할 목록은 `manifest.csv`. 원본 출처 기록은 `data/raw/trashcan_1_0/provenance.json`.

실제 재분할: train 4,918장(218개 영상), val 1,159장(47개 영상), test 1,135장(47개 영상). 분할 간 영상 ID 교집합은 모두 0개입니다. 7,212장 전체를 사용하며 검토 제외/동일 픽셀 중복 제거는 0장입니다. bbox 보정은 3,286개입니다.

val 원본 `vid_000143_frame0000013.jpg`의 ROV annotation 두 개는 polygon이 다르지만 category와 bbox가 완전히 같았습니다. detection 변환 시 동일 클래스/좌표의 정답 박스 한 개만 유지하고 `label_duplicates.csv`에 중복 한 개 제거를 기록했습니다. 원본 segmentation/JSON과 이미지는 변경하지 않았습니다. 최종 detection 정답은 12,127개입니다.

학습은 config.yaml의 YOLO26n, 입력 640, AdamW, batch 8, seed 42, 모델별 50 epoch를 사용합니다. A/B는 같은 원본 분할/학습 횟수/기본 증강 조건을 사용하며 B 학습 이미지만 추가 수중 증강합니다. 테스트 데이터는 학습과 모델 선택에 사용하지 않습니다.

실행 흐름은 `src.run_research`: A 학습 → A 독립 test 평가 → B 학습 → B 독립 test 평가 → 비교 그래프. 단계마다 `results/research_status.json`을 갱신하고 오류 발생 시 failed를 기록합니다. 현재 상태는 문서의 고정 문장 대신 해당 JSON과 실제 프로세스/로그에서 확인하세요.

2026-10-12 06:29 KST에 전체 검증 통과 후 A의 1/50 epoch에서 실제 optimizer 학습 배치 진행과 손실 기록을 확인했습니다. A/B 전체 완료를 의미하지 않습니다. 독립된 숨김 백그라운드 프로세스가 실행하며 stdout/stderr는 `results/trashcan_training_stdout.log`, `results/trashcan_training_stderr.log`에 저장합니다. 최초 중복 라벨 검증 실패 기록은 `results/failed_runs/duplicate_bbox_20261012`에 보존했습니다. 전처리/경계 보정/중복 라벨/실행 단계에 대한 자동 검사 8개와 코드 문법 검사를 통과했습니다.

최종 모델 경로: `models/A_seed42/weights/best.pt`, `models/B_seed42/weights/best.pt`. 완료 후 실제 평가값은 `results/evaluation/A_seed42`, `results/evaluation/B_seed42`와 `results/comparison`에 저장됩니다. 아직 완료되지 않은 실행의 성능 수치를 만들어 기록하지 않습니다.

이 학습 환경의 PyTorch는 CPU 빌드이고 CUDA가 감지되지 않았습니다. 장시간 작업은 컴퓨터가 켜져 있어야 합니다. 실패/전원 종료 후 완성되지 않은 run 폴더를 지워 재시작하지 말고 로그와 체크포인트를 먼저 보존하세요.

2026-10-15 후속 확인에서 원래 PID 63776이 사라졌고 다른 학습 프로세스도 없음을 확인했습니다. 오류 로그에 종료 원인은 남아 있지 않았습니다. `last.pt`는 완료 epoch 18과 optimizer 상태를 포함하고, epochs 50/imgsz 640/batch 8/seed 42 조건이 유지돼 있음을 실제 파일에서 확인했습니다. `results/recoveries/20261015_process_exit`에 이전 가중치·CSV·설정·로그를 복사하고 `src.run_research --resume`로 복구 실행했습니다. 미완료 epoch 19를 다시 시작하며 목표는 여전히 모델별 총 50 epoch입니다. 재개 관련 검사 3개와 기존 전처리 검사 7개를 통과했습니다. 이 실행은 중단·재개된 실험으로 보고서에도 해당 이력을 포함해야 하며, 중단 없는 실행과 비트 단위 동일 결과를 보장하지 않습니다. 실제 재개 성공/진행 상태는 최신 로그와 상태 JSON을 확인하세요.

사용 조건: 학술 교육/연구용 무료이며 저자 Hong, Jungseok; Fulton, Michael S; Sattar, Junaed 및 원본 이미지 제공자 JAMSTEC을 보고서/발표 자료에 표기해야 합니다. 공식 LICENSE.txt 원문을 보관했습니다.

복구 확인: 원본 7,212장 동일성 검사를 다시 통과했고, 2026-10-15 05:14 KST에 실제 19/50 epoch 배치가 여러 번 진행되며 손실을 기록하는 것을 확인했습니다. 재개 체크포인트 SHA256은 `945df81fe06ac69eb5177dc6c6ee316fef936635a2d5cedd9cae63512730e68a`입니다. 전체 완료나 최종 test 성능을 의미하지 않습니다.

2026-10-15 두 번째 복구: 시스템 부팅 시각이 같은 날 14:30 KST로 바뀌었으며 이전 PID 82132와 다른 학습 프로세스가 없음을 확인했습니다. `last.pt`를 직접 열어 완료 epoch 26, optimizer 상태, 총 50 epoch/imgsz 640/batch 8/seed 42를 확인했습니다. `results/recoveries/20261015_reboot_recovery`에 재부팅 전 체크포인트·CSV·설정·로그와 `recovery.json`을 보존하고, 18:17 KST에 `src.run_research --resume`를 다시 실행했습니다. 미완료 epoch 27부터 재개하며 목표 epoch나 데이터를 줄이지 않았습니다. 재개 진행은 최신 상태 JSON과 로그로 확인하세요.

2026-10-16 A 학습/평가 완료: `models/A_seed42/experiment.json`과 `results/evaluation/A_seed42/evaluation.json`의 completed 상태를 확인하고, 실제 `best.pt` SHA256이 양쪽 기록과 모두 일치함을 검증했습니다. A는 총 50 epoch를 마쳤으며 독립 test 1,135장/정답 2,089개를 평가했습니다. conf 0.25/매칭 IoU 0.5에서 micro Precision 49.66%, Recall 42.03%, F1 45.53%; 낮은 AP cutoff 0.001의 native mAP50 40.84%, mAP50-95 24.72%입니다. 실제 원본 수치는 `results/evaluation/A_seed42/summary.csv`와 `per_class.csv`, 그림은 같은 폴더의 `confusion_matrix.png`와 모델 폴더의 `results.png`입니다. Marine_Life Recall은 16.67%로 미탐이 많습니다. 이를 완성된 실용 성능이나 B의 개선 증거로 해석하지 않으며, 미리 정한 B 조건과 동일 test로 비교해야 합니다.

같은 날 후속 확인에서 B는 1 epoch 완료 후 PID 266804가 사라졌고 다른 학습도 없음을 확인했습니다. stderr는 비어 있고 추가 시스템 재부팅은 확인되지 않아 정확한 종료 원인은 알 수 없습니다. B의 optimizer가 있는 `last.pt`와 기존 50/640/8/42 설정을 실제 파일에서 확인하고 `results/recoveries/20261016_B_process_exit`에 이전 B 가중치/로그/설정 및 완료된 A 평가 기록 사본을 보존했습니다. `src.run_research --resume`로 완료된 A 결과를 재사용하고 B의 기존 수중 증강 데이터에서 2/50 epoch부터 이어가도록 실행했습니다. A 결과나 데이터/epoch/증강 조건은 변경하지 않았습니다.

같은 날 15:56 KST에 B의 실제 2/50 epoch 배치 진행을 확인했습니다. 혼동행렬은 오분류 진단을 위한 클래스 무관 IoU 매칭이며, 표의 P/R/F1은 클래스 일치 IoU 매칭으로 계산하므로 TP 집계가 서로 다를 수 있습니다. B가 미완료이므로 A/B 개선 여부에 대한 결론은 아직 내리지 않습니다.

2026-10-17 B 재복구: 후속 확인에서 PID 257324가 사라졌고 다른 학습도 없음을 확인했습니다. 추가 시스템 재부팅이나 확인한 최근 Python crash 이벤트는 없고 stderr도 비어 있어 종료 원인은 확정하지 않았습니다. B의 `last.pt`를 열어 완료 epoch 27과 optimizer 상태, 동일 50/640/8/42 설정을 확인했습니다. `results/recoveries/20261017_B_process_exit`에 이전 가중치·CSV·로그·설정을 보존하고 `src.run_research --resume`로 미완료 epoch 28부터 복구 실행했습니다. A의 완료된 학습/평가는 재사용하며 기존 수중 증강 데이터도 유지합니다. 최신 실제 진행은 상태 JSON과 로그를 확인하세요.
