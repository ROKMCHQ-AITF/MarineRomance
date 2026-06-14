# 오디오 분류 대회 실전 비법서

> 숙련된 ML 엔지니어가 비전공 대학생에게 주는 압축본.
> 읽는 것보다 **순서대로 따라가는 게** 목적이다.

---

## 0. 대회 첫날 — 30분 안에 끝낼 것

```
□ 대회 Overview 탭 전부 읽기 (평가지표가 뭔지 반드시 확인)
□ Data 탭: 파일 형식(wav/ogg/flac), 샘플레이트, 클래스 수, 레이블 구조 확인
□ Discussion 탭: 상위권 공개 노트북, 데이터 quirk, 리크 이슈 체크
□ Leaderboard: public/private 비율, shake-up 주의보 여부 확인
□ input/ 에 데이터 다운로드 후 verify_audio.py 실행
□ prepare_folds.py 실행 → folds.csv 생성
□ exp001_baseline.yaml의 num_classes 업데이트
□ python main.py --config configs/exp001_baseline.yaml wandb.mode=disabled 실행
   → loss 떨어지면 OK. 여기서 막히면 다른 거 하지 말 것.
```

**첫날 목표는 하나: 학습이 돌아가는 것.**

---

## 1. EDA — 뭘 봐야 하나

### 반드시 확인하는 것

| 항목 | 왜 중요한가 | 확인 방법 |
|------|------------|---------|
| **클래스 분포** | 불균형 심하면 loss 전략 바뀜 | `df['label'].value_counts()` |
| **오디오 길이 분포** | 너무 짧은 파일 다수면 padding 전략 필요 | `librosa.get_duration()` 히스토그램 |
| **샘플레이트 통일 여부** | 혼재하면 전처리 버그 원인 | `verify_audio.py` 결과 |
| **손상 파일** | 조용히 0 텐서로 들어가 학습 오염 | `verify_audio.py` |
| **멀티레이블 여부** | 한 클립에 여러 소리? → BCE 필수 | 레이블 컬럼 `split` 후 분포 확인 |
| **배경 소음 비율** | nocall/background 클래스 있으면 따로 처리 | 파형 시각화 5~10개 |

### 보면 좋은 것

```python
# 01_eda.ipynb에서 실행
import librosa, librosa.display, matplotlib.pyplot as plt

# 파형 + 멜스펙 동시 시각화
wav, sr = librosa.load("sample.wav", sr=32000)
mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

fig, axes = plt.subplots(2, 1)
librosa.display.waveshow(wav, sr=sr, ax=axes[0])
librosa.display.specshow(mel_db, sr=sr, ax=axes[1], x_axis='time', y_axis='mel')
```

**클래스별로 3~5개씩** 들어보고 봐라. 귀가 가장 빠른 EDA 도구다.

---

## 2. Baseline 세팅 체크리스트

```
□ sample_rate: 대부분 32000 or 22050. 원본이 뭔지 확인 후 맞추거나 낮춰도 됨
□ duration: 5초가 무난. 클래스당 평균 길이 보고 판단
□ n_mels: 128 시작, 나중에 64/256 실험
□ image_size: [128,128] baseline → [224,224] or [256,256] 점수 오르면 올리기
□ backbone: tf_efficientnet_b0_ns (빠르고 강함, 상위권 1번 선택)
□ loss: multilabel=true면 bce, single-label이면 ce
□ metric: 대회 공식 지표로 반드시 맞출 것 (f1인지 auc인지 cmap인지)
□ folds: 1-fold로 빠른 실험 → 나중에 5-fold
□ epochs: 10~15로 시작, LR curve 보고 조정
```

---

## 3. 상위권이 하는 것 — 압축 핵심

### 3-1. 스코어 올리는 순서 (효과 큰 순)

```
1. 데이터 품질 개선         ← 가장 크다. 손상 파일 제거, 노이즈 클립 제거
2. Duration 최적화          ← 5초 vs 10초 vs 30초 실험
3. Backbone 업그레이드      ← B0 → B2/B4 → EfficientNetV2-M
4. 멜스펙 파라미터 튜닝     ← n_mels 64/128/256, fmax 대회별 최적치 다름
5. 증강 추가                ← SpecAugment, Mixup, 배경음 합성
6. Loss 개선                ← Focal Loss (클래스 불균형), Label Smoothing
7. 5-fold 앙상블            ← 안정적인 +0.005~0.01
8. 후처리 (threshold 조정)  ← OOF로 최적 threshold 탐색
```

> **함정:** 2~3번부터 시작하는 초보가 많다. 데이터 품질이 먼저다.

### 3-2. 오디오 대회 고유 패턴

**배경음 합성 증강 (BirdCLEF 상위권 필수)**
```python
# 배경 노이즈 클립을 따로 모아서 주 오디오에 섞음
# 대회에서 보통 nocall/background 데이터를 제공함
mixed = primary_wav * 0.8 + background_wav * 0.2
```

**Secondary label 활용**
```python
# BirdCLEF 등에서 secondary_labels 컬럼이 있으면 무시하지 말 것
# soft label로 변환해서 쓰면 +0.01~0.02
y[secondary_idx] = 0.3   # hard 1.0 대신 soft
```

**Sliding window inference**
```python
# 테스트 오디오가 길 때 (30초짜리 클립 등)
# 5초씩 슬라이딩해서 예측 후 max pooling
# train: 5초, test: window로 쪼개서 max → 훨씬 좋음
```

**Rating/quality 필터링**
```python
# 대회 메타데이터에 rating 컬럼 있으면 low-quality 제거
df = df[df['rating'] >= 3.0]   # 상위권이 자주 씀
```

### 3-3. SpecAugment 세팅 기준

```yaml
# 음향 대회 기준 경험치
augment:
  spec_augment: true
  freq_mask: 24    # n_mels의 15~20%
  time_mask: 40    # 타임 프레임의 10~15%
```

### 3-4. 앙상블 전략

```
Level 1: 같은 구조, 다른 fold seed → 5개 평균 (+0.005)
Level 2: 다른 backbone (B0 + B2 + B4) → +0.01~0.02
Level 3: 다른 feature (mel + mfcc) → +0.005~0.01
Level 4: Stacking (OOF로 meta-learner 훈련) → 드라마틱하진 않음

→ 마감 2~3일 전까지 Level 2까지만 해도 상위권
```

---

## 4. 실험 관리 — 상위권은 이렇게 한다

### config 파일 네이밍 원칙

```
exp001_baseline.yaml      ← B0, 1-fold, 5초, 기본
exp002_b2_specaug.yaml    ← B2 + SpecAugment
exp003_focal_mixup.yaml   ← Focal loss + Mixup
exp004_5fold.yaml         ← 5-fold 앙상블용
```

### wandb 사용 원칙

```python
# 실험할 때는 online, 디버깅할 때는 disabled
wandb.mode: online    # 실험 로깅
wandb.mode: disabled  # 디버깅/테스트

# 반드시 태그 달기
wandb.tags: [b2, specaug, fold0]
```

### 뭘 로깅해야 하나

```
필수: train_loss, val_loss, val_[metric], lr
선택: grad_norm (발산 감지), epoch_time (속도 추적)
```

---

## 5. 단계별 작업 플로우

### Phase 1: 기반 세팅 (Day 1~2)

```bash
# 1. 데이터 검증
python scripts/verify_audio.py --config configs/exp001_baseline.yaml

# 2. Fold 생성
python scripts/prepare_folds.py --config configs/exp001_baseline.yaml

# 3. Baseline 학습 (wandb 끄고 빠르게)
python main.py --config configs/exp001_baseline.yaml \
  train.epochs=3 wandb.mode=disabled

# 4. 학습 확인 후 전체 epoch 학습
python main.py --config configs/exp001_baseline.yaml
```

**Phase 1 체크:**
```
□ loss가 떨어지는가?
□ val metric이 0 이상인가? (0.0이면 metric 계산 버그)
□ 체크포인트가 outputs/에 저장되는가?
□ OOM 없이 돌아가는가?
```

---

### Phase 2: 빠른 실험 (Day 3~7)

한 번에 하나만 바꾼다. 동시에 두 개 바꾸면 뭐 때문인지 모른다.

```bash
# B2로 업그레이드
python main.py --config configs/exp001_baseline.yaml \
  model.backbone=tf_efficientnet_b2_ns exp_name=exp002_b2

# SpecAugment 추가
python main.py --config configs/exp001_baseline.yaml \
  augment.spec_augment=true exp_name=exp003_specaug

# image_size 키우기
python main.py --config configs/exp001_baseline.yaml \
  feature.image_size=[224,224] exp_name=exp004_224
```

**실험 기록 템플릿:**

| exp | 변경점 | val_f1 | 비고 |
|-----|--------|--------|------|
| 001 | baseline B0 | 0.72 | |
| 002 | B2 | 0.74 | +0.02 채택 |
| 003 | SpecAugment | 0.73 | 미채택 |

---

### Phase 3: 스코어 짜내기 (Day 8~마감 3일 전)

```bash
# 5-fold 전체 학습
bash scripts/train_all_folds.sh configs/exp002_best.yaml

# OOF 기반 threshold 탐색 (직접 구현)
python -c "
import numpy as np
from sklearn.metrics import f1_score

oof_preds = np.load('outputs/exp002/oof_fold0.npy')  # concat all folds
oof_labels = np.load('outputs/exp002/oof_labels.npy')

best_thr, best_score = 0.5, 0.0
for thr in np.linspace(0.1, 0.9, 80):
    score = f1_score(oof_labels, oof_preds > thr, average='samples', zero_division=0)
    if score > best_score:
        best_score, best_thr = score, thr
print(f'Best threshold: {best_thr:.3f}, score: {best_score:.4f}')
"

# 추론 + 앙상블
python inference.py \
  --config configs/exp002_best.yaml \
  --ckpt outputs/exp002_best \
  --threshold 0.35    # OOF에서 찾은 최적값
```

---

### Phase 4: 마감 3일 전

```
□ Public LB score와 OOF score의 상관관계 확인
  → 괴리가 크면 fold 설계 문제 or 데이터 리크
□ 제출 2개 전략 선택:
  1. 가장 높은 public LB 모델
  2. OOF 기반으로 가장 신뢰하는 앙상블 모델
  → 둘 다 제출. private에서는 2번이 더 안전한 경우가 많음
□ 코드 재현성 최종 확인: seed 고정 + requirements.txt
```

---

## 6. 자주 하는 실수 — 피해라

### 데이터 관련

| 실수 | 결과 | 방지법 |
|------|------|--------|
| 손상 파일 그대로 학습 | 학습 오염, 점수 하락 | `verify_audio.py` 필수 |
| train/val에 같은 그룹 혼재 | OOF 점수 뻥튀기, 실제 낮음 | `group_col` 설정 확인 |
| 레이블 인코딩 불일치 | submission 완전 망 | `label_map.json` 검증 |
| test set 전처리 누락 | train/test 분포 불일치 | Dataset mode='test' 확인 |

### 학습 관련

| 실수 | 결과 | 방지법 |
|------|------|--------|
| LR 너무 높음 | loss NaN | `1e-4` 시작, 천천히 올리기 |
| 배치 너무 작음 | BatchNorm 불안정 | 최소 16, 가능하면 32 |
| epoch 너무 적음 | underfitting | LR curve 보고 판단 |
| 검증셋 없이 epoch 고정 | 실전에서 과적합 | 반드시 validation 봐라 |

### 제출 관련

| 실수 | 결과 | 방지법 |
|------|------|--------|
| threshold=0.5 그냥 제출 | 점수 손해 0.02~0.05 | OOF로 최적화 |
| 단일 fold만 제출 | 분산 큼 | 마감 전엔 5-fold |
| sample_submission 포맷 무시 | 오류 제출 | 컬럼명/타입 꼭 맞추기 |

---

## 7. 음향 대회 용어 빠른 참조

| 용어 | 뜻 | 이 코드에서 |
|------|-----|------------|
| **mel spectrogram** | 주파수를 인간 청각 스케일로 변환한 2D 이미지 | `frontend.py` |
| **SpecAugment** | mel spec에서 주파수/시간 축 랜덤 마스킹 | `augment.SpecAugment` |
| **Mixup** | 두 샘플을 λ 비율로 섞어 soft label 생성 | `augment.mixup_batch` |
| **OOF (Out-of-Fold)** | 학습에 안 쓴 fold의 validation 예측 | `trainer.py` val 결과 |
| **cMAP** | 클래스별 Average Precision 평균 (음향 대회 흔함) | `metrics._cmap_score` |
| **SED** | Sound Event Detection, 타임스탬프 단위 예측 | `heads.py` sed head |
| **nocall** | 배경음만 있고 타겟 소리 없는 클립 | 클래스로 처리하거나 별도 필터링 |
| **frontend** | waveform → spectrogram GPU 레이어 | `models/frontend.py` |

---

## 8. 긴급 디버깅 체크리스트

### val_metric = 0.0 계속 나올 때

```
1. metric.name이 대회 지표랑 맞는가?
2. 멀티레이블인데 sigmoid 안 쓰고 softmax 쓰는 건 아닌가?
3. 레이블이 전부 0으로 들어가는 건 아닌가?
   → dataset에서 y[:5] 출력해서 확인
4. threshold가 너무 높아서 전부 negative인 건 아닌가?
   → 예측 확률 분포 출력: print(preds.mean(), preds.max())
```

### Loss가 NaN이 될 때

```
1. LR을 1/10으로 줄여라
2. AMP 끄고 돌려봐라 (train.amp=false)
3. 입력 텐서에 NaN/Inf 있는지 확인
   → torch.isnan(x).any() 체크
4. 손상 파일이 원인인 경우 많음 → verify_audio.py 재실행
```

### OOM (Out of Memory) 날 때

```
1. batch_size 절반으로
2. train.grad_accum=2 로 effective batch size 유지
3. feature.image_size 줄이기 [128, 128]
4. num_workers 줄이기 (핀 메모리도 메모리 씀)
```

---

## 9. 이 레포 파일 지도

```
configs/exp001_baseline.yaml   ← 매 실험마다 복사해서 수정
src/data/dataset.py            ← 데이터 안 들어오면 여기 먼저
src/models/frontend.py         ← spectrogram 모양 바꾸려면 여기
src/models/backbones.py        ← backbone 추가하려면 여기
src/training/trainer.py        ← 학습 루프 커스텀하려면 여기
src/utils/metrics.py           ← 대회 지표 추가하려면 여기
main.py                        ← K-fold 오케스트레이션만, 건드릴 일 거의 없음
inference.py                   ← 제출 파일 생성, threshold 조정
```

---

## 10. 한 줄 요약

> 데이터 검증 → 학습 돌아가게 → 하나씩 실험 → 앙상블 → threshold 최적화.
> 순서가 틀리면 시간만 버린다.
