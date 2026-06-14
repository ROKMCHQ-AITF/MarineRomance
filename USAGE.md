# USAGE.md — 사용 설명서

> 사람이 읽는 매뉴얼. "새 대회를 시작했다 → 제출 파일을 만든다"까지의 실제 절차.
> 구현 디테일은 `IMPLEMENTATION.md`, 코드 구조는 `ARCHITECTURE.md` 참고.

---

## 0. 5분 요약

```bash
# 1. 환경
pip install -r requirements.txt
wandb login                       # 처음 1회

# 2. 데이터를 input/ 에 둔다 (train_metadata.csv, train_audio/ 등)

# 3. 대회에 맞게 config를 연다 (아래 "새 대회 체크리스트" 참고)
vim configs/exp001_baseline.yaml

# 4. 준비 → 학습 → 추론
python scripts/verify_audio.py  --config configs/exp001_baseline.yaml
python scripts/prepare_folds.py --config configs/exp001_baseline.yaml
python main.py                  --config configs/exp001_baseline.yaml
python inference.py             --config configs/exp001_baseline.yaml --ckpt outputs/exp001_baseline
```

`exp001_baseline`은 1-fold·가벼운 모델로 "파이프라인이 돈다"를 확인하는 용도다.
점수를 내려면 `exp002_advanced`(5-fold·무거운 모델·mixup)로 간다.

---

## 1. 설치

```bash
git clone <repo>
cd kaggle_audio_clf
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU/CUDA 버전에 맞는 PyTorch는 별도 설치가 필요할 수 있다.
wandb는 `wandb login` 한 번이면 된다. 안 쓸 거면 config의 `wandb.mode: disabled`.

---

## 2. 데이터 배치

```
input/
├── train_metadata.csv     # id, label, (group) 컬럼
├── train_audio/           # 학습 오디오
└── test/                  # 테스트 오디오 (추론 시)
```
대회마다 컬럼명이 다르므로 config의 `data.label_col`, `data.id_col`,
`data.group_col`을 실제 CSV에 맞춘다.

---

## 3. 새 대회 시작 체크리스트 ⭐

새 음향 분류 대회를 받으면 **config만 손보면** 대부분 끝난다. 순서대로:

1. **`input/`에 데이터 배치** (위 §2)
2. **`configs/exp001_baseline.yaml`에서 아래를 대회에 맞게 수정**
   - `data.sample_rate` — 데이터의 실제 SR (예: 32000)
   - `data.duration` — 한 클립 길이(초). 짧은 소리면 5s, 긴 soundscape면 더 길게/슬라이딩
   - `data.label_col` / `data.id_col` / `data.group_col` — CSV 컬럼명
   - `data.multilabel` — 한 클립에 여러 라벨이면 true
   - `model.num_classes` — (또는 prepare_folds가 출력해주는 값을 넣는다)
   - `metric.name` — **대회 공식 평가지표로 교체** (F1/AUC/cmAP…)
   - `wandb.project` — 대회 이름
3. **`scripts/verify_audio.py` 실행** — 깨진 파일·SR 불일치 먼저 잡는다
4. **`utils/metrics.py`에 공식 지표 확인/구현** — 리더보드와 로컬 CV가 같은 값을 보게
5. **`scripts/prepare_folds.py` 실행** — fold 나누고 `num_classes` 확인
6. **`python main.py --config configs/exp001_baseline.yaml debug=true`** — 1% 데이터로 5분 내 e2e 통과 확인
7. 통과하면 `debug` 빼고 본 학습. 점수 나오면 `exp002_advanced`로 확장

> 대부분의 대회는 1~7로 baseline 제출까지 간다. feature·model 튜닝은 그 다음.

---

## 4. 주요 명령어

### 학습
```bash
# 단일 fold (빠른 검증)
python main.py --config configs/exp001_baseline.yaml

# 전체 5-fold (백그라운드)
nohup bash scripts/train_all_folds.sh configs/exp002_advanced.yaml > train.log 2>&1 &

# CLI로 yaml 값 덮어쓰기 (yaml을 안 고치고 빠르게 실험)
python main.py --config configs/exp002_advanced.yaml \
    train.epochs=20 optimizer.lr=5e-4 augment.mixup=0.5 train.folds=[0,1]
```

### 추론·제출
```bash
# 단일 체크포인트
python inference.py --config configs/exp002_advanced.yaml \
    --ckpt outputs/exp002_advanced/best_fold0.pth

# 디렉토리 지정 시 best_fold*.pth 전부 앙상블
python inference.py --config configs/exp002_advanced.yaml \
    --ckpt outputs/exp002_advanced --threshold 0.5

# 쉘 스크립트 래퍼 (exp_name 자동 감지)
bash scripts/make_submission.sh configs/exp002_advanced.yaml
```

### 보조 스크립트
```bash
python scripts/verify_audio.py  --config <cfg>   # 데이터 무결성
python scripts/prepare_folds.py --config <cfg>   # fold 생성
python scripts/cache_features.py --config <cfg>  # feature 캐싱(I/O 가속)
```

---

## 5. 자주 바꾸는 설정 빠른 참조

| 목적 | 바꾸는 키 | 예시 값 |
|---|---|---|
| 특징추출 종류 | `feature.type` | `melspec` / `mfcc` / `cqt` / `raw` |
| 3채널 구성 | `feature.channel_mode` | `repeat` / `delta` / `multi_res` |
| 멜 해상도 | `feature.n_mels`, `feature.image_size` | `128`, `[256,256]` |
| 모델 | `model.backbone` | `tf_efficientnet_b0_ns`, `convnext_small` |
| 음향 사전학습 | `model.type` | `panns` / `wav2vec2` |
| 학습 길이 | `train.epochs`, `train.batch_size` | |
| mixup | `augment.mixup` | `0.5` |
| SpecAugment | `augment.spec_augment` | `true` |
| loss | `loss.type` | `bce` / `focal` / `lsep` |
| 스케줄러 | `optimizer.scheduler` | `cosine` / `onecycle` |
| 슬라이딩 추론 | `inference.sliding_window` | `true` (긴 오디오) |
| 디버그 모드 | `debug` | `true` (1% 데이터, 1 epoch) |
| wandb 끄기 | `wandb.mode` | `disabled` |

---

## 6. 산출물 위치

```
outputs/<exp_name>/
├── config.yaml          # 실행 시점 config (재현용)
├── best_fold{n}.pth     # fold별 best 가중치
├── oof_fold{n}.npy      # out-of-fold 예측 (임계값 최적화·앙상블용)
├── submission.csv       # inference.py 결과
└── cache/               # cache_features.py 결과 (feature.compute_on=cpu 시)
    └── <stem>.npy
```
wandb 대시보드에는 loss·metric 곡선, lr 스케줄, (옵션) 오디오 샘플이 올라간다.

---

## 7. 추천 워크플로우 (대회 전체 흐름)

1. **Day 1** — `exp001_baseline`로 파이프라인 e2e 통과 + 첫 제출(점수가 낮아도 리더보드에 올린다)
2. **EDA** — `notebooks/01_eda.ipynb`로 클래스 분포·duration·SNR 확인
3. **로컬 CV 신뢰 구축** — `metrics.py`가 대회 공식 지표와 일치하는지, CV-LB 상관 확인
4. **feature 탐색** — `feature.type`/`channel_mode`/`n_mels`를 바꿔 몇 개 실험
5. **model 탐색** — backbone 몇 개 비교 (가벼운 것부터)
6. **증강·loss 튜닝** — mixup, SpecAugment, focal/lsep
7. **5-fold 확장** — `exp002_advanced`로 풀 학습
8. **후처리** — `postprocess/threshold.py`로 임계값 최적화
9. **앙상블·TTA** — 다양한 fold/backbone 평균, 슬라이딩 윈도우
10. **(후반) pseudo-labeling** — 상위권 단골 기법. 자신감 높은 예측을 학습에 추가

---

## 8. 트러블슈팅

| 증상 | 점검 |
|---|---|
| 학습이 안 도는데 import 에러 | 스텁이 다 채워졌는지. `debug=true`로 e2e부터 |
| GPU OOM | `train.batch_size`↓, `feature.image_size`↓, `train.grad_accum`↑ |
| DataLoader가 느림(GPU 놀음) | `cache_features.py`로 feature 캐싱, `num_workers`↑ |
| 로컬 CV는 좋은데 LB가 나쁨 | fold 누수(group_col 확인), CV-LB 지표 불일치, 도메인 시프트 |
| wandb 때문에 멈춤 | `wandb.mode=disabled`로 우회 후 logger 래퍼 확인 |
| 라벨 매핑 오류 | `prepare_folds.py`가 만든 `label_map.json`과 num_classes 일치 확인 |
| 깨진 오디오로 크래시 | `verify_audio.py` 먼저 돌려 문제 파일 제외 |

---

## 9. Claude Code와 같이 작업할 때

- 새 기능을 시킬 땐 **"`IMPLEMENTATION.md`의 X 모듈을 채워줘"** 식으로 모듈을 지정한다.
- 큰 변경 전엔 "`CLAUDE.md`의 불변 규칙을 지켜서"를 덧붙이면 설계가 흐트러지지 않는다.
- "config에 키를 추가하지 말고 기존 스키마 안에서" 같은 제약을 주면 config 난립을 막는다.
- 처음 스캐폴딩은 **"구현 순서 1~7까지만, 1차 마일스톤까지"**로 끊어서 시킨다.
