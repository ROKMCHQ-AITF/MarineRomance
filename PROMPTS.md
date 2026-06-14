# PROMPTS.md — 대회 상황별 Claude Code 프롬프트 모음

> 대회 당일, 상황이 생길 때마다 해당 섹션을 열고 프롬프트를 복붙한다.
> 모든 프롬프트는 **CLAUDE.md의 불변 규칙을 지키면서** 동작하도록 설계됐다.
> `<꺾쇠>` 안의 값만 실제 값으로 바꿔서 쓴다.

---

## 📋 목차

- [0. 대회 시작 직후 — 데이터 파악](#0-대회-시작-직후)
- [1. config 세팅](#1-config-세팅)
- [2. 구현 채우기](#2-구현-채우기)
- [3. Feature 실험](#3-feature-실험)
- [4. Model 실험](#4-model-실험)
- [5. 증강 실험](#5-증강-실험)
- [6. 평가지표 대응](#6-평가지표-대응)
- [7. 디버깅](#7-디버깅)
- [8. 점수 올리기 (후반)](#8-점수-올리기-후반)
- [9. 제출 직전](#9-제출-직전)

---

## 0. 대회 시작 직후

### 0-1. 데이터 파악 (가장 먼저)
```
train_metadata.csv를 열어서 아래를 파악해줘:
- 라벨 컬럼명이 뭔지
- 클래스가 몇 개인지
- 한 샘플에 라벨이 하나인지 여러 개인지 (single/multi label)
- id 컬럼명이 뭔지
- group으로 쓸 수 있는 컬럼이 있는지 (author, location, recording_id 등)

오디오 파일 하나를 로드해서:
- sample_rate가 몇인지
- 평균 duration이 몇 초인지
- 파일 포맷이 뭔지 (wav/mp3/ogg)

결과를 표로 정리해줘.
```

### 0-2. EDA 빠르게
```
notebooks/01_eda.ipynb를 실행해서 아래를 확인하고 요약해줘:
- 클래스 분포 (불균형 정도)
- duration 분포 (짧은 클립인지 긴 soundscape인지)
- 샘플 수가 적은 클래스 top 10
- 무음/노이즈가 많은지

이걸 바탕으로 당장 주의해야 할 점을 3줄로 알려줘.
```

---

## 1. Config 세팅

### 1-1. 대회 데이터에 맞게 config 업데이트
```
아래 정보를 바탕으로 configs/exp001_baseline.yaml을 업데이트해줘.
IMPLEMENTATION.md의 config 스키마를 기준으로 맞는 키에 값을 넣어.

- 라벨 컬럼명: <label_col>
- id 컬럼명: <id_col>
- 클래스 수: <num_classes>
- multilabel 여부: <true/false>
- sample_rate: <sr>
- 오디오 평균 길이: <duration>초
- 평가지표: <metric_name>
- group 컬럼: <있으면 컬럼명, 없으면 null>
```

### 1-2. 새 실험 config 만들기
```
configs/exp001_baseline.yaml을 베이스로
configs/<exp_name>.yaml을 새로 만들어줘.

바꾸고 싶은 것:
- <예: feature.type을 mfcc로>
- <예: model.backbone을 convnext_small로>
- <예: train.epochs를 30으로>

나머지는 baseline과 동일하게 상속.
```

---

## 2. 구현 채우기

### 2-1. 전체 스텁 → 실제 구현 (1차 마일스톤)
```
CLAUDE.md의 구현 순서 1~7을 따라서,
현재 NotImplementedError 상태인 스텁들을 실제 동작하는 코드로 채워줘.

목표: python main.py --config configs/exp001_baseline.yaml debug=true 가
에러 없이 끝까지 실행되는 것.

조건:
- CLAUDE.md의 불변 규칙 7개를 지킬 것
- IMPLEMENTATION.md의 함수 시그니처를 그대로 따를 것
- 한 번에 다 짜지 말고, 모듈 하나 짤 때마다 import 에러 없는지 확인하면서 진행
```

### 2-2. 특정 모듈만 구현
```
IMPLEMENTATION.md를 읽고 <모듈명>을 구현해줘.

현재 상황:
- <예: dataset.py는 됐고, dataloader.py가 NotImplementedError>

조건:
- 함수 시그니처는 IMPLEMENTATION.md 것 그대로
- shape 주석 달 것 (B, C, H, W 등)
- wandb 호출은 utils/logger.py 래퍼를 통해서만
```

### 2-3. prepare_folds 실행 전 구현
```
scripts/prepare_folds.py를 구현해줘.

데이터 상황:
- CSV 경로: input/train_metadata.csv
- 라벨 컬럼: <label_col>
- multilabel: <true/false>
- group 컬럼: <있으면 컬럼명>
- n_folds: 5

요구사항:
- multilabel=true면 iterative-stratification 사용 (iterstrat 라이브러리)
- group 컬럼 있으면 GroupKFold
- 결과를 input/folds.csv로 저장
- label_map.json도 같이 생성 (클래스명 → 인덱스)
- 실행 후 클래스 수를 출력해서 config에 반영할 수 있게
```

---

## 3. Feature 실험

### 3-1. Feature 종류 교체
```
현재 feature.type=melspec인데 <mfcc/cqt/raw>로 바꿔서 실험하고 싶어.

models/frontend.py에 <mfcc/cqt/raw> 분기를 추가해줘.
data/feature_extractor.py에도 동일한 분기 추가.

ARCHITECTURE.md의 "이걸 바꾸려면 어디를 여나" 표를 참고해서,
두 파일 외에 건드릴 곳이 있으면 같이 알려줘.

추가 후 configs/exp_<feature>.yaml도 만들어줘.
```

### 3-2. 3채널 구성 방식 실험
```
현재 feature.channel_mode=repeat인데 delta와 multi_res도 실험하고 싶어.

models/frontend.py의 channel_mode 분기에 아래를 추가해줘:
- delta: [mel, delta(mel), delta-delta(mel)] 스택
- multi_res: n_fft를 [512, 1024, 2048]로 달리한 3개 mel 스택
  (image_size로 리사이즈해서 맞춤)

각각 configs/exp_delta.yaml, configs/exp_multires.yaml도 만들어줘.
```

### 3-3. 스펙트로그램 파라미터 튜닝
```
현재 mel 파라미터가 n_mels=128, image_size=[256,256]인데
아래 조합으로 실험 config를 만들어줘:

A: n_mels=64,  image_size=[224,224]  → 가벼운 baseline
B: n_mels=128, image_size=[256,256]  → 현재
C: n_mels=256, image_size=[384,384]  → 고해상도

파일명: configs/exp_mel_A.yaml, B.yaml, C.yaml
backbone은 전부 tf_efficientnet_b0_ns로 동일하게.
```

---

## 4. Model 실험

### 4-1. Backbone 교체
```
아래 backbone들로 각각 실험 config를 만들어줘.
models/backbones.py에서 지원 여부 확인하고, 안 되는 게 있으면 추가해줘.

- tf_efficientnet_b0_ns   (가벼움, 빠름)
- tf_efficientnet_b4_ns   (중간)
- convnext_small          (최신 CNN)
- eca_nfnet_l0            (상위권 단골)
- tf_efficientnetv2_s     (속도·성능 균형)

각각 configs/exp_<backbone이름>.yaml으로.
batch_size는 모델 크기에 맞게 조정해줘 (b0=64, b4=32, 나머지=32).
```

### 4-2. 음향 사전학습 모델 연결
```
models/pretrained_audio.py에 PANNs(CNN14)를 연결해줘.

참고:
- PANNs 공식 repo: https://github.com/qiuqiangkong/audioset_tagging_cnn
- 가중치는 input/pretrained/Cnn14_mAP=0.431.pth 에 있다고 가정
- factory.py에서 model.type=panns일 때 이걸 불러오도록
- frontend는 건너뛰고 waveform을 바로 받는 구조로

configs/exp_panns.yaml도 만들어줘.
```

### 4-3. Head 교체 (Attention Pooling)
```
현재 model.head=linear인데 attention pooling으로 바꾸고 싶어.

models/heads.py의 AttentionHead를 구현해줘.
PANNs식으로: clipwise_output = attention_weighted_sum(framewise_output)

configs/exp_attn_head.yaml도 만들어줘 (model.head=attention).
```

---

## 5. 증강 실험

### 5-1. SpecAugment 켜기
```
augment.spec_augment=true로 설정하고
data/augment.py의 SpecAugment 클래스를 구현해줘.

파라미터:
- freq_mask_param: cfg.augment.freq_mask (기본 24)
- time_mask_param: cfg.augment.time_mask (기본 40)
- 마스크 개수: freq 2개, time 2개

training/trainer.py의 train_one_epoch에서
frontend 통과 직후 (스펙트로그램 나온 다음) 적용되도록.
```

### 5-2. Mixup 켜기
```
augment.mixup=0.5로 설정하고 mixup을 활성화해줘.

data/augment.py의 mixup_batch 함수 구현:
- alpha=cfg.augment.mixup로 Beta 분포에서 lambda 샘플링
- multilabel이므로 mixup_mode=max (두 라벨의 max)
- x_mixed = lam*x1 + (1-lam)*x2
- y_mixed = max(y1, y2)

trainer.py의 train_one_epoch에서 GPU 스텝 안에서 호출.
loss 계산 시 mixed y를 쓰도록.
```

### 5-3. Waveform 증강 추가
```
data/augment.py의 apply_waveform_aug에 아래를 추가해줘:
- gain: 랜덤 볼륨 조절 (0.5~1.5배)
- noise: 가우시안 노이즈 추가 (SNR 20~40dB)
- time_shift: 랜덤 circular shift (duration의 10% 이내)

각각 cfg.augment.gain/noise/time_shift=true일 때만 적용.
dataset.py의 __getitem__에서 train mode일 때만 호출.
```

---

## 6. 평가지표 대응

### 6-1. 대회 공식 지표 구현
```
utils/metrics.py에 아래 지표를 구현해줘:
지표명: <F1 / macro_AUC / cmAP / padded_cmAP / lwlrap>

구현 조건:
- 함수 시그니처: (y_true: np.ndarray, y_prob: np.ndarray) -> float
- y_true shape: (N, num_classes), multi-hot
- y_prob shape: (N, num_classes), sigmoid 통과 전 logit 또는 후 확률 (명시해줘)
- get_metric_fn(cfg)가 이 함수를 반환하도록

configs의 metric.name: <지표명>도 업데이트.
```

### 6-2. 임계값 최적화
```
postprocess/threshold.py를 구현해줘.

oof 예측(outputs/<exp>/oof_fold*.npy)과
oof 정답(outputs/<exp>/oof_labels.npy)을 읽어서:

1. global threshold 탐색 (0.1~0.9 grid)
2. per-class threshold 탐색
3. 최적 threshold에서의 metric 값 출력
4. 결과를 outputs/<exp>/thresholds.json으로 저장

평가지표: cfg.metric.name 사용.
```

---

## 7. 디버깅

### 7-1. 에러 발생 시
```
아래 에러가 발생했어:

<에러 메시지 전체 붙여넣기>

ARCHITECTURE.md의 데이터 흐름과 모듈 의존성을 참고해서
원인이 어디인지 찾고 고쳐줘.
고칠 때 CLAUDE.md의 불변 규칙을 깨지 않는지 확인해줘.
```

### 7-2. Shape 에러
```
아래 shape 에러가 났어:

<에러>

ARCHITECTURE.md §5 텐서 shape 규약을 기준으로
어느 모듈에서 shape가 틀렸는지 추적해줘.
각 모듈 경계에서 실제 shape를 print해서 확인하는 디버그 코드도 추가해줘.
```

### 7-3. 학습이 이상할 때 (loss가 안 줄거나 NaN)
```
학습 중 아래 증상이 있어:
<예: loss가 NaN / val_metric이 0으로 고정 / train loss는 줄지만 val이 안 줄음>

아래를 순서대로 체크해줘:
1. label이 제대로 multi-hot으로 인코딩됐는지
2. loss 함수 입력이 logit인지 sigmoid 통과값인지 (BCEWithLogits는 logit)
3. metric 계산 시 threshold 적용 여부
4. data leakage (fold 분할 확인)
5. lr이 너무 크거나 작은지
```

### 7-4. 속도가 느릴 때
```
학습이 너무 느려 (GPU 사용률이 낮음).
아래를 확인하고 병목을 찾아줘:

1. DataLoader num_workers가 적절한지
2. pin_memory 설정
3. feature를 매번 계산하는지 (cache_features.py로 캐싱 필요한지)
4. AMP(자동 혼합 정밀도)가 켜져 있는지
5. image_size가 너무 큰지

개선안을 config 또는 코드 수정으로 제안해줘.
```

---

## 8. 점수 올리기 (후반)

### 8-1. 5-fold 전체 학습
```
configs/exp002_advanced.yaml을 만들어줘.

exp001_baseline 대비 바꿀 것:
- train.folds: [0,1,2,3,4]  (전체 fold)
- train.epochs: 30
- augment.mixup: 0.5
- augment.spec_augment: true
- model.backbone: <지금까지 실험에서 가장 좋았던 것>
- feature.type: <가장 좋았던 것>
- optimizer.scheduler: cosine
- train.ema: true  (있으면)

그리고 scripts/train_all_folds.sh도 이 config로 업데이트.
```

### 8-2. 앙상블
```
inference.py에 앙상블 기능을 추가해줘.

앙상블할 가중치:
<outputs/exp_A/fold0_best.pth, fold1_best.pth, ...>
<outputs/exp_B/fold0_best.pth, ...>

방식:
- 각 모델의 sigmoid 확률을 평균
- (선택) 가중 평균: 각 fold의 val_metric을 가중치로

thresholds.json이 있으면 apply_threshold 적용 후 제출.
```

### 8-3. TTA (Test Time Augmentation)
```
inference.py에 음향 특화 TTA를 추가해줘.

cfg.inference.tta 목록:
- none: 원본 그대로
- time_shift: 오디오를 0.5초 shift한 버전 추가 추론
- speed_change: 속도를 0.9/1.1배로 바꾼 버전 추가 추론

각 TTA 결과의 sigmoid 확률을 평균해서 최종 예측으로.
configs에 inference.tta: [none, time_shift] 추가.
```

### 8-4. Pseudo Labeling
```
pseudo label 파이프라인을 만들어줘.

순서:
1. 현재 best 모델로 unlabeled 데이터 추론
   (입력: input/unlabeled/, 출력: outputs/pseudo_labels.csv)
2. 확률이 <threshold=0.7> 이상인 예측만 hard label로 채택
3. pseudo label을 train 데이터에 추가한 새 CSV 생성
   (outputs/train_with_pseudo.csv)
4. 이걸로 재학습하는 configs/exp_pseudo.yaml 생성
   (data.train_csv를 새 CSV로 교체)

CLAUDE.md의 코딩 컨벤션 따를 것.
```

---

## 9. 제출 직전

### 9-1. 제출 파일 생성
```
inference.py를 실행해서 submission.csv를 만들어줘.

설정:
- config: configs/exp002_advanced.yaml
- 가중치: outputs/exp002_advanced/ 아래 전체 fold best
- TTA: [none, time_shift]
- threshold: outputs/exp002_advanced/thresholds.json

submission.csv 포맷이 대회 요구사항과 맞는지 확인해줘:
- 컬럼명: <id_col>, <label 또는 확률 컬럼>
- 행 수: test 샘플 수와 동일한지
```

### 9-2. 재현성 체크
```
이번 실험을 나중에 그대로 재현할 수 있는지 확인해줘.

체크리스트:
1. outputs/<exp>/config.yaml이 저장됐는지
2. requirements.txt가 현재 환경과 일치하는지 (pip freeze > requirements.txt)
3. seed가 모든 진입점에서 seed_everything()으로 고정됐는지
4. wandb에 run이 기록됐는지 (run url 출력)
5. 가중치 파일이 전부 있는지

문제 있는 항목 알려줘.
```

---

## 🔧 자주 쓰는 디버그 원라이너

```bash
# 파이프라인 최소 실행 (1% 데이터, 1 epoch, wandb 끔)
python main.py --config configs/exp001_baseline.yaml debug=true wandb.mode=disabled

# fold 하나만, epoch 2개, 빠른 확인
python main.py --config configs/exp002_advanced.yaml train.folds=[0] train.epochs=2

# wandb 끄고 로컬 로그만
python main.py --config configs/exp001_baseline.yaml wandb.mode=disabled

# feature 캐싱 후 학습 (I/O 느릴 때)
python scripts/cache_features.py --config configs/exp001_baseline.yaml
python main.py --config configs/exp001_baseline.yaml feature.compute_on=cpu
```

---

## 📌 프롬프트 쓸 때 주의사항

1. **항상 CLAUDE.md 불변 규칙 언급** — "CLAUDE.md의 불변 규칙을 지키면서"를 붙이면 설계가 안 흐트러진다.
2. **모듈 단위로 끊어서** — 한 번에 너무 많이 시키면 Claude Code가 설계를 임의로 결정한다.
3. **시그니처 변경 금지** — "IMPLEMENTATION.md의 함수 시그니처 그대로"를 붙인다.
4. **에러는 전문 붙여넣기** — 요약하지 말고 에러 메시지 전체를 그대로.
5. **실험 결과는 기록** — wandb 또는 메모에 config + val_metric을 남긴다. 어떤 실험이 좋았는지 기억 못 하면 시간 낭비.
