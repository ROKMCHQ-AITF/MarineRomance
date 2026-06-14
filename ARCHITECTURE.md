# ARCHITECTURE.md

> **코드를 읽을 때 보는 지도.** "이 데이터가 어디서 와서 어디로 가는가", "이 모듈을 고치면
> 어디가 영향받는가"를 빠르게 파악하기 위한 문서. 구현 스펙은 `IMPLEMENTATION.md`에 있다.

---

## 1. 한 장으로 보는 데이터 흐름

```
                     [ CPU 영역 (src/data) ]            [ GPU 영역 (src/models) ]
                ┌─────────────────────────────┐   ┌──────────────────────────────────┐
  folds.csv ──► dataset.py                     │   │                                  │
  (한 행)       ├─ preprocessing.load_audio    │   │   factory.AudioModel             │
                │   리샘플·모노·정규화          │   │   ┌────────────────────────────┐ │
  WAV 파일 ───► ├─ fix_length (crop)           │   │   │ frontend.py                │ │
                ├─ augment.apply_waveform_aug  │   │   │  STFT/Mel/MFCC/CQT (GPU)   │ │
                │   (gain/noise/shift)         │   │   │  to_db·normalize·resize    │ │
                └─ label → multi-hot           │   │   │  channel_mode → C채널       │ │
                          │                    │   │   └────────────┬───────────────┘ │
                          ▼                    │   │                ▼                  │
                   dataloader.py               │   │   SpecAugment (GPU, train만)      │
                   배치 묶기 ───── waveform ────┼───┼──►            │                  │
                   (B,1,T) + label             │   │                ▼                  │
                └─────────────────────────────┘   │   backbones.py (timm)             │
                                                   │     또는 pretrained_audio.py       │
                                                   │                │                  │
                                                   │                ▼                  │
                                                   │   heads.py (linear/sed/attention) │
                                                   │                │                  │
                                                   │                ▼   logits         │
                                                   └────────────────┼──────────────────┘
                                                                    ▼
                                          training/trainer.py:  mixup → loss → backward → step
                                                                    │
                                                  metrics.py ◄──────┴─────► checkpoint.py
                                                                    │
                                                            utils/logger.py (wandb)
```

핵심: **경계는 dataloader다.** dataloader 왼쪽은 전부 CPU(waveform까지), 오른쪽은 전부 GPU(feature부터).
스펙트로그램은 GPU에서 모델 안에서 만들어진다. 이걸 기억하면 "어디서 뭘 고쳐야 하는지" 80%가 풀린다.

> 예외 경로: `cfg.feature.compute_on == cpu`면 `cache_features.py`가 미리 만든 `.npy` feature를
> dataset이 직접 로드하고, factory의 frontend는 건너뛴다. 출력 규격은 GPU 경로와 동일.

---

## 2. 진입점에서부터 따라가기

### 학습: `main.py`
```
main.py
 ├─ load_config()              # configs/default + 실험 yaml 병합
 ├─ seed_everything()
 ├─ Logger(cfg)                # wandb 초기화 (disabled면 no-op)
 ├─ read folds.csv
 └─ for fold in cfg.train.folds:
       ├─ train/valid df 분리 (df.fold 기준)
       ├─ build_dataloader() x2
       ├─ build_model()        # ← factory.py, feature·model 조립
       └─ Trainer(...).fit()   # ← 한 fold 학습 전체
            ├─ train_one_epoch()  # AMP·mixup·specaug·loss·step
            ├─ validate()         # metric, oof 저장
            └─ BestTracker        # best일 때 checkpoint 저장
```
`main.py`는 **fold를 돌리는 일만** 한다. 실제 학습 루프는 `Trainer` 안에 있다.

### 추론: `inference.py`
```
inference.py --config <yaml> --ckpt <dir_or_.pth>
 ├─ load_config()
 ├─ _find_checkpoints(--ckpt)   # 단일 .pth 또는 best_fold*.pth 목록
 ├─ test dataloader (mode='test')
 ├─ 각 체크포인트: build_model + load state_dict
 │    └─ predict_one_model(): TTA(none/flip/gain_up/gain_down) 평균
 ├─ 모델 앙상블 단순 평균
 ├─ threshold 적용 + label_map.json 으로 역매핑
 └─ submission.csv 저장 (id_col + 'prediction')
```

---

## 3. "이걸 바꾸려면 어디를 여나" 빠른 색인

| 하고 싶은 것 | 여는 파일 | 비고 |
|---|---|---|
| feature를 mel→cqt로 변경 | `configs/*.yaml`의 `feature.type` | ⚠️ cqt는 frontend에 NotImplementedError — 구현 필요 |
| 새 feature 종류 추가 | `models/frontend.py` + `data/feature_extractor.py` | feature_extractor.py 미구현 — 두 곳 동시 추가 |
| backbone 교체 | `configs/*.yaml`의 `model.backbone` | timm 이름만 바꾸면 됨 |
| 음향 사전학습 모델(PANNs 등) | `models/pretrained_audio.py` + `model.type` | ❌ 파일 미구현 — 새로 작성 필요 |
| 3채널 구성 방식 | `feature.channel_mode` (repeat/delta/multi_res) | ⚠️ repeat만 동작. delta/multi_res는 NotImplementedError |
| 증강 켜고 끄기 | `configs/*.yaml`의 `augment.*` | waveform=dataset, spec=trainer |
| mixup | `augment.mixup` + `trainer.train_one_epoch` | GPU 스텝에서 적용 |
| loss 변경 | `loss.type` → `training/losses.py` | ⚠️ bce/focal/ce 동작. lsep는 NotImplementedError |
| 스케줄러 변경 | `optimizer.scheduler` → `training/optimizers.py` | |
| 평가지표(대회 공식) | `metric.name` → `utils/metrics.py` | f1/macro_f1/auc/cmap 지원 |
| fold 분할 방식 | `scripts/prepare_folds.py` | Stratified/Group/MultilabelStratified |
| SED(이벤트 탐지)로 전환 | `model.head=sed` → `heads.SEDHead` | ⚠️ SEDHead는 NotImplementedError |
| TTA·앙상블 | `inference.py` | none/gain TTA 동작. flip/sliding_window 미구현 |
| 임계값 최적화 | `postprocess/threshold.py` | ❌ 파일 미구현 — 새로 작성 필요 |
| CPU feature 캐싱 | `scripts/cache_features.py` | ❌ 파일 미구현 — feature_extractor.py와 함께 작성 |
| 학습 로그 항목 추가 | `utils/logger.py` 호출부(trainer) | wandb |

---

## 4. 모듈 의존성 (누가 누구를 import 하나)

```
main.py
 ├─ utils.config, utils.seed, utils.logger
 ├─ data.dataloader ─► data.dataset ─► data.preprocessing, data.augment
 ├─ models.factory ─► models.frontend, models.backbones,
 │                    models.pretrained_audio [❌ 미구현], models.heads
 └─ training.trainer ─► training.losses, training.optimizers,
                        data.augment(mixup/specaug),
                        utils.metrics, utils.checkpoint, utils.logger

inference.py
 ├─ utils.config, utils.seed
 ├─ data.dataset, data.dataloader
 ├─ models.factory
 └─ (postprocess.threshold [❌ 미구현] — OOF threshold 최적화 시 별도 호출)

scripts/prepare_folds.py   ─► (독립) pandas + sklearn
scripts/verify_audio.py    ─► data.preprocessing
scripts/cache_features.py  [❌ 미구현] ─► data.feature_extractor [❌ 미구현]
```

규칙: **의존성은 한 방향으로만 흐른다.** `utils`는 아무것도 import 안 하고(가장 아래),
`data`/`models`/`training`은 `utils`만 본다. 진입점(`main`/`inference`)이 전부를 조립한다.
순환 import가 생기면 설계가 틀린 것.

---

## 5. 텐서 shape 규약 (머릿속에 고정)

| 단계 | 변수 | shape | 비고 |
|---|---|---|---|
| dataset 출력 | wav | `(1, T)` | T = sample_rate × duration |
| dataloader 출력 | batch wav | `(B, 1, T)` | |
| frontend 출력 | feature | `(B, C, H, W)` | C=n_channels, H≈n_mels, W=시간프레임 |
| backbone 출력 | feat | `(B, D)` | D=feature_dim |
| head 출력 | logits | `(B, num_classes)` | SED면 추가로 `(B, T', num_classes)` |
| label | y | `(B, num_classes)` | multi-hot (다중레이블 기본) |

shape 주석을 코드에 항상 달아둔다. 음향 파이프라인 버그의 절반은 shape에서 난다.

---

## 6. config가 코드를 지배하는 방식

이 레포는 "config-driven"이다. 코드는 **분기 로직만** 갖고, **값은 전부 config**에서 온다.
그래서 코드를 읽을 때는 항상 **해당 실험의 yaml을 같이 펴놓고** 본다.

```
configs/exp001_baseline.yaml  ← "지금 무슨 설정으로 도는가"의 단일 진실
        │
        ├─ feature.type=melspec   → frontend.py의 melspec 분기가 실행됨
        ├─ model.backbone=...     → backbones.py가 그 모델을 만듦
        └─ loss.type=bce          → losses.py가 BCE를 반환
```

`outputs/<exp>/config.yaml`에 **실행 시점 config가 덤프**되므로, 과거 실험을 재현·디버그할 땐
그 파일을 본다(코드가 아니라).

---

## 7. 읽는 순서 추천

1. 이 문서 §1 데이터 흐름도 — 큰 그림
2. `configs/exp001_baseline.yaml` — 무슨 설정인지
3. `main.py` — 오케스트레이션
4. `models/factory.py` → `frontend.py` — 음향 파이프라인의 심장
5. `training/trainer.py` — 학습 스텝 디테일
6. 나머지는 §3 색인을 보고 필요할 때 점프
