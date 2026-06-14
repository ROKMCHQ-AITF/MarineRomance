# CLAUDE.md

> 이 파일은 Claude Code가 이 레포에서 작업할 때 **항상 먼저 읽는** 지침서다.
> 새 세션을 시작하면 이 파일 → `ARCHITECTURE.md` → 작업 대상 모듈 순으로 읽는다.

---

## 1. 이 프로젝트가 뭔가

Kaggle **Audio Classification** 대회용 baseline 템플릿이다.
목표는 "대회 페이지를 연 그날 바로 `python main.py --config configs/exp001_baseline.yaml`로
1-fold 학습이 돌아가는" 수준의 골격을 만드는 것이다.

핵심 설계 철학은 단 하나로 요약된다:

> **특징추출법(feature)과 모델(model)을 config의 한 줄로 갈아끼울 수 있어야 한다.**

아직 어떤 feature(mel / mfcc / cqt / raw)와 어떤 backbone(EfficientNet / ConvNeXt / PANNs / wav2vec2)을
쓸지 정해지지 않았다. 그래서 코드는 **특정 feature·model에 종속되면 안 되고**, 전부
config 값으로 분기되는 팩토리 패턴이어야 한다. 이게 이 레포의 가장 중요한 불변 규칙이다.

---

## 2. 기술 스택 (고정)

- PyTorch (+ AMP)
- `timm` — 이미지 backbone
- `torchaudio` / `nnAudio` — 스펙트로그램 변환 (GPU 레이어)
- `librosa` — CPU 전처리·EDA용
- `Weights & Biases (wandb)` — 실험 로깅 (**이 프로젝트는 wandb를 쓴다. tensorboard는 옵션**)
- `pandas` / `numpy` / `scikit-learn` — fold 분할, 메타데이터
- `PyYAML` / `OmegaConf` — config

버전은 `requirements.txt`에 고정한다. 재현성이 점수다.

---

## 3. 디렉토리 구조

```
kaggle_audio_clf/
├── configs/
│   ├── default.yaml            # 모든 실험이 상속하는 공통 베이스
│   ├── exp001_baseline.yaml    # 가벼운 검증용: 1ch, 가벼운 모델, 1-fold
│   └── exp002_advanced.yaml    # 스코어링용: 3ch fusion, 무거운 모델, 5-fold, mixup
│
├── input/                      # 데이터 (gitignore). 대회 데이터를 여기에 둔다
├── outputs/                    # 가중치/oof/로그 (gitignore)
│
├── notebooks/
│   ├── 01_eda.ipynb            # 파형·스펙트럼·클래스 분포·duration 분포
│   └── 02_check_features.ipynb # feature_extractor 결과 눈으로 검증
│
├── src/
│   ├── data/                   # CPU 담당: 디스크 I/O + 전처리
│   │   ├── dataset.py          # 메타데이터 파싱, raw 오디오 로드, crop
│   │   ├── dataloader.py       # DataLoader + num_workers + collate
│   │   ├── preprocessing.py    # [신규] 리샘플링/정규화/무음제거(VAD)/DC offset
│   │   ├── feature_extractor.py# [신규] CPU 경로 특징추출 (캐싱용 mel/mfcc/cqt)
│   │   └── augment.py          # waveform + spectrogram 증강 (SpecAugment 등)
│   │
│   ├── models/                 # GPU 담당: 신경망
│   │   ├── frontend.py         # [핵심] waveform→스펙트로그램→3ch 텐서 (GPU 레이어)
│   │   ├── backbones.py        # timm 연동
│   │   ├── pretrained_audio.py # [신규] PANNs/wav2vec2/BirdNET 등 음향 사전학습
│   │   ├── heads.py            # 분류/회귀/다중레이블/SED 출력층
│   │   └── factory.py          # [신규] config→model 조립 (frontend+backbone+head)
│   │
│   ├── training/
│   │   ├── trainer.py          # epoch 단위 fit/eval, AMP, mixup 적용 지점
│   │   ├── losses.py           # BCE / Focal / CE / LSEP
│   │   └── optimizers.py       # optimizer + scheduler (cosine, warmup)
│   │
│   ├── postprocess/            # [신규]
│   │   └── threshold.py        # 임계값 최적화, per-class threshold, 스무딩
│   │
│   └── utils/
│       ├── config.py           # [신규] yaml 로드 + default 병합 + CLI override
│       ├── metrics.py          # 대회 공식 지표 (F1/AUC/cmAP 등)
│       ├── logger.py           # wandb 연동
│       ├── checkpoint.py       # best 가중치 저장/로드
│       └── seed.py             # [신규] 전역 시드 고정
│
├── scripts/
│   ├── prepare_folds.py        # [신규] StratifiedKFold/GroupKFold → folds.csv 생성
│   ├── verify_audio.py         # [신규] 손상파일/SR 불일치/무음 검출
│   ├── cache_features.py       # [신규] mel 등을 .npy로 미리 캐싱 (I/O 병목 제거)
│   ├── train_all_folds.sh      # 백그라운드 무인 5-fold 학습
│   └── make_submission.sh      # 추론 → submission.csv
│
├── main.py                     # K-Fold 학습 진입점
├── inference.py                # 테스트 추론 + TTA + 앙상블
├── requirements.txt
├── CLAUDE.md                   # ← 지금 이 파일
├── IMPLEMENTATION.md           # 각 모듈 구현 스펙 (Claude Code 작업 지시서)
├── ARCHITECTURE.md             # 코드 읽기 가이드 (데이터 흐름·모듈 관계)
└── USAGE.md                    # 사용 설명서 (사람이 읽는 매뉴얼)
```

`[신규]`/`[핵심]` 표시는 원본 템플릿에 없던, **반드시 새로 구현해야 하는** 파일이다.

---

## 4. 절대 어기면 안 되는 아키텍처 불변 규칙 (Invariants)

1. **Config가 진실의 원천.** 하드코딩된 하이퍼파라미터 금지. `n_mels`, `lr`, `backbone` 이름 등
   모든 값은 yaml에서 온다. 코드에 `n_mels=128` 같은 매직넘버를 박지 마라.

2. **Feature와 Model은 팩토리로 조립한다.** `src/models/factory.py`의 `build_model(cfg)`가
   `cfg.feature.type` / `cfg.model.type`을 보고 frontend·backbone·head를 조립한다.
   새 feature/model을 추가할 때 **기존 코드 수정 없이** 분기 한 줄만 추가하면 되도록 짠다.

3. **스펙트로그램은 GPU에서 계산하는 것을 기본 경로로 한다.** (TheoViel/상위권 패턴)
   `frontend.py`는 `nn.Module`이고 `forward(waveform) -> image_tensor`를 한다.
   이렇게 하면 batch 단위로 GPU에서 변환 + spectrogram 도메인 증강이 가능하다.
   CPU 캐싱 경로(`cache_features.py`)는 대안일 뿐, 둘 다 같은 인터페이스를 따른다.

4. **CPU 일과 GPU 일을 섞지 마라.** `src/data/*`는 CPU(디스크 I/O, waveform 전처리),
   `src/models/*`는 GPU(스펙트로그램·신경망). DataLoader는 waveform 텐서까지만 넘긴다.

5. **다중레이블을 기본 가정으로 둔다.** 음향 분류는 multi-label인 경우가 많다(한 클립에 여러 소리).
   loss/metric/head는 single-label과 multi-label을 config로 전환 가능하게 짠다.
   기본은 BCEWithLogits + multi-label.

6. **재현성.** 모든 진입점은 `utils/seed.py`의 `seed_everything(cfg.seed)`를 가장 먼저 호출한다.

7. **wandb는 끌 수 있어야 한다.** `cfg.wandb.mode`가 `disabled`면 wandb 호출이 전부 no-op이 되도록
   `utils/logger.py`에서 래핑한다. 디버깅 중 wandb 때문에 멈추면 안 된다.

---

## 5. 코딩 컨벤션

- 타입힌트 필수. public 함수는 docstring 1~3줄.
- 함수는 짧게. 한 함수가 한 가지 일만.
- 파일 상단에 그 파일의 책임을 1줄 주석으로.
- import 순서: 표준 → 서드파티 → 로컬(`from src.xxx import`).
- 경로는 `pathlib.Path`. 문자열 경로 연결(`+ "/"`) 금지.
- 텐서 shape를 주석으로 명시: `# (B, C, H, W)`.
- 예외를 삼키지 마라. `verify_audio.py` 외에는 `try/except: pass` 금지.
- 새 의존성을 추가하면 즉시 `requirements.txt`에 버전 고정해서 반영.

---

## 6. 자주 쓰는 명령어

```bash
# 0) 데이터 검증 (학습 전 1회)
python scripts/verify_audio.py --config configs/exp001_baseline.yaml

# 1) fold 분할 (1회)
python scripts/prepare_folds.py --config configs/exp001_baseline.yaml

# 2) (선택) feature 캐싱 — 미구현. cache_features.py + feature_extractor.py 작성 후 사용 가능
# python scripts/cache_features.py --config configs/exp002_advanced.yaml

# 3) 학습 (단일 fold, 빠른 검증)
python main.py --config configs/exp001_baseline.yaml

# 3') 전체 fold 백그라운드 학습
bash scripts/train_all_folds.sh configs/exp002_advanced.yaml

# 4) 추론 + 제출 파일 생성
python inference.py --config configs/exp002_advanced.yaml --ckpt outputs/exp002_advanced
bash scripts/make_submission.sh

# CLI override (yaml 값을 명령행에서 덮어쓰기)
python main.py --config configs/exp001_baseline.yaml train.epochs=2 train.batch_size=8 wandb.mode=disabled
```

---

## 7. 구현 순서 (현재 상태)

새 기능 추가 시 이 순서 원칙을 따른다.

1. ✅ `utils/config.py`, `utils/seed.py`, `utils/logger.py` — 기반
2. ✅ `configs/default.yaml` + `exp001_baseline.yaml` — config 스키마 확정
3. ✅ `data/dataset.py`, `data/dataloader.py` — 더미 텐서라도 batch가 나오게
4. ✅ `models/frontend.py`, `models/backbones.py`, `models/heads.py`, `models/factory.py`
   - ⚠️ frontend: `melspec + repeat` 채널만 동작. `mfcc/cqt/raw`, `delta/multi_res`는 `NotImplementedError`
   - ⚠️ heads: `LinearHead`만 동작. `AttentionHead`, `SEDHead`는 `NotImplementedError`
5. ✅ `training/losses.py`, `training/optimizers.py`, `training/trainer.py`
   - ⚠️ losses: `bce/focal/ce`만 동작. `lsep`는 `NotImplementedError`
6. ✅ `utils/metrics.py`, `utils/checkpoint.py`
7. ✅ `main.py` — 1-fold 학습 e2e 통과 (1차 마일스톤 달성)
8. ✅ `scripts/prepare_folds.py`, `verify_audio.py`
9. ✅ `data/augment.py` (mixup/specaug), `data/preprocessing.py`
10. ✅ `inference.py` + fold 앙상블
11. ❌ `data/feature_extractor.py`, `scripts/cache_features.py` — 미구현 (CPU 캐싱 경로)
12. ❌ `models/pretrained_audio.py` — 미구현 (PANNs/wav2vec2 등)
    ❌ `postprocess/threshold.py` — 미구현 (임계값 최적화)

**코어 파이프라인(1~10) 완료. 실제 대회 데이터 투입 준비 완료.**
`input/`에 데이터 넣고 config 수정 후 바로 학습 가능.
11~12는 성능 향상 고도화 단계에서 추가 구현한다.

---

## 8. 하지 말 것

- ❌ 특정 대회(BirdCLEF 등) 클래스 수·라벨을 코드에 하드코딩 — 전부 config·메타데이터에서.
- ❌ feature 계산을 dataset과 model 양쪽에 중복 구현 — frontend(GPU) 또는 cache(CPU) 중 하나의 경로로.
- ❌ `main.py`에 학습 루프 본문을 직접 작성 — 루프는 `trainer.py`에, `main.py`는 K-Fold 오케스트레이션만.
- ❌ notebook에 재사용 로직 작성 — 로직은 `src/`에, notebook은 호출·시각화만.
- ❌ 검증 안 된 무거운 의존성 추가 — 새 라이브러리는 먼저 물어본다.
- ❌ 1차 마일스톤 전에 pseudo-label·앙상블 같은 고급 기능부터 만들기.

---

## 9. 작업할 때 참고 파일

- 무엇을 구현하는가 → **`IMPLEMENTATION.md`** (모듈별 함수 시그니처·인터페이스)
- 코드가 어떻게 연결되는가 → **`ARCHITECTURE.md`** (데이터 흐름도)
- 사람이 어떻게 쓰는가 → **`USAGE.md`**
