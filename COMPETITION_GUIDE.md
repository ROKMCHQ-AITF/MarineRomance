# COMPETITION_GUIDE.md
# 무박 2일 Audio Classification 대회 완전 공략서

> 상위권 솔루션 분석 기반. 개념 설명 + 실전 전략을 같이 담는다.
> "왜 이걸 하는가"를 이해하면 처음 보는 데이터에서도 유연하게 응용할 수 있다.

---

## ⏱️ 무박 2일 타임라인 (먼저 보기)

```
[0h ~2h]   대회 파악 · EDA · CV 구축 · 첫 제출
[2h ~8h]   feature / model 탐색 · 빠른 실험
[8h ~16h]  5-fold 풀 학습 · 증강 추가
[16h~24h]  앙상블 · threshold 최적화 · pseudo label
[24h~48h]  마무리 · 재현성 확인 · 최종 제출 선택
```

---

## 0. 대회 시작 직후 — 상위권의 첫 2시간

상위권이 대회 첫날 하는 행동은 거의 공식처럼 정해져 있다.

### 0-1. 데이터 파악 (30분)

```
체크리스트:
□ train_metadata.csv 열어서 컬럼 구조 파악
□ 클래스 수, multilabel 여부
□ 클래스 분포 (불균형 정도 — 심하면 전략이 달라진다)
□ 오디오 1개 로드해서 sample_rate / duration 확인
□ 파일 포맷 (wav / mp3 / ogg)
□ 무음·노이즈 비율 (대충 10개 들어봄)
□ test 데이터 구조 (short clip인지 long soundscape인지)
```

**왜 중요한가:**
- 클래스 불균형이 심하면 → loss에 class_weights 또는 focal loss
- test가 long soundscape이면 → sliding window 추론 필수
- duration이 짧으면(1~3초) → 그대로, 길면(>10초) → random crop 전략

### 0-2. CV 전략 확정 (30분)

**개념: Cross-Validation이 전부다**

대회에서 가장 중요한 단일 결정이 CV 전략이다.
CV가 흔들리면 어떤 실험이 좋은지 알 수 없고, 그러면 아무것도 못 한다.

```
상위권 원칙:
"CV가 LB와 상관관계가 확인되면, 이후엔 CV만 믿는다."
"LB를 과하게 제출해서 overfitting하는 것보다 CV를 믿는 게 낫다."
```

| 상황 | CV 전략 |
|---|---|
| 일반적인 경우 | StratifiedKFold 5-fold |
| 한 명이 여러 녹음을 냈을 때 | GroupKFold (group=author/recorder_id) |
| 시간 순서가 있을 때 | TimeSeriesSplit |
| multilabel + 불균형 | iterative-stratification |

**CV-LB 상관 확인 방법:**
1. fold0 하나로 학습 → 제출 → LB 점수 기록
2. 로컬 CV와 LB 점수 차이 확인
3. 차이가 일정하면 OK → 이후엔 CV만 보고 실험

### 0-3. 첫 제출 (1시간)

```
목표: 점수가 아니라 "파이프라인이 도는지" 확인
- 1-fold, 가벼운 모델, 5 epoch으로 제출
- LB 점수 확인 (어느 수준인지 감 잡기)
- 이게 baseline. 이후 모든 실험은 여기서 얼마나 올라가는지가 기준
```

---

## 1. 모델 선정 — 범용 SOTA

### 개념: 왜 이미지 모델을 음향에 쓰는가?

오디오를 mel-spectrogram으로 변환하면 2D 이미지가 된다.
x축=시간, y축=주파수, 밝기=에너지. 이미지 분류 모델을 그대로 쓸 수 있다.
ImageNet으로 사전학습된 모델이 texture·pattern을 잘 잡는다는 게 실험적으로 증명됐다.

### 상위권 backbone 선택 기준

**우선순위: 속도·정확도 균형 → 다양성 확보**

| Backbone | 특징 | 언제 쓰나 |
|---|---|---|
| `tf_efficientnet_b0_ns` | 가볍고 빠름. 음향 대회 단골 1위 | baseline, 추론 제약 있을 때 |
| `tf_efficientnet_b4_ns` | b0보다 정확하지만 느림 | 추론 시간 여유 있을 때 |
| `eca_nfnet_l0` | 상위권에서 자주 나옴 | 앙상블 다양성 |
| `convnext_small` | 최신 CNN, 안정적 | 다양성 확보 |
| `tf_efficientnetv2_s` | v2 시리즈, 빠르고 강함 | 중간 크기 필요할 때 |
| `seresnext50_32x4d` | ResNet 계열 강자 | 앙상블 다양성 |

**핵심 전략:**
```
1단계: b0으로 파이프라인 검증 (빠름)
2단계: b4 / nfnet_l0 / convnext 중 1~2개 추가
3단계: 다른 계열 backbone들로 앙상블 다양성 확보
```

**왜 b0_ns인가:** `_ns` = NoisyStudent로 학습됨. 일반 b0보다 robustness가 높다.
음향 데이터는 노이즈가 많아서 ns 버전이 잘 맞는다.

### PANNs (음향 특화 사전학습)

**개념:** ImageNet 대신 AudioSet(유튜브 200만개 오디오)으로 사전학습된 모델.
음향 도메인에 훨씬 맞는 가중치에서 시작한다.

```python
# 대표 모델
CNN14       # PANNs 대표. 강력하지만 무거움
MobileNetV1 # PANNs 경량 버전
```

**언제 쓰나:** 데이터가 적거나, 음향 도메인이 특수할 때 (새소리, 환경음 등).
데이터가 충분하면 ImageNet pretrained + mel이 더 빠르게 수렴한다.

---

## 2. 학습 전략

### 2-1. 기본 학습 설정

**상위권 표준 세팅 (BirdCLEF 기준):**

```yaml
optimizer: AdamW
lr: 1e-3 ~ 5e-4
scheduler: CosineAnnealingLR (warmup 포함)
warmup_ratio: 0.1   # 전체 step의 10%를 warmup
epochs: 15~30
batch_size: 32~64
amp: true           # 자동 혼합 정밀도 (필수)
```

**개념: Cosine Annealing + Warmup**
```
학습률이 처음엔 낮게 시작 (warmup) → 최대로 올라감 → cos 곡선으로 서서히 감소
처음에 lr이 크면 사전학습 가중치가 망가진다. warmup이 이걸 방지함.
```

### 2-2. Loss 전략

| Loss | 언제 쓰나 | 개념 |
|---|---|---|
| `BCEWithLogitsLoss` | multilabel 기본 | 클래스 독립적으로 이진분류 |
| `Focal Loss` | 클래스 불균형 심할 때 | 쉬운 샘플 가중치↓, 어려운 샘플 가중치↑ |
| `CE` | single-label | 일반 분류 |
| `LSEP` | ranking 기반 지표(AUC)일 때 | 상위권 단골 |

**Focal Loss 개념:**
```
일반 BCE: 모든 샘플을 똑같이 대우
Focal:    맞추기 쉬운 샘플(확률 높은 것)의 gradient를 줄여서
          어렵고 소수인 클래스에 집중하게 만듦
gamma=2.0이 표준. 불균형이 심할수록 gamma를 크게.
```

### 2-3. 2단계 학습 전략 (상위권 단골)

```
Stage 1: backbone frozen → head만 학습 (5 epoch)
         → 빠르게 head를 안정화
Stage 2: 전체 unfreeze → 낮은 lr로 fine-tuning
         (backbone lr = head lr / 10)

왜: 사전학습 가중치를 처음부터 너무 빠르게 바꾸면 망가짐
    head를 먼저 안정화하고 전체를 천천히 조정
```

### 2-4. EMA (Exponential Moving Average)

**개념:** 학습 중 가중치의 이동평균을 별도로 관리.
실제 가중치보다 더 안정적이고 일반화가 좋다.
```
ema_weight = decay * ema_weight + (1 - decay) * model_weight
decay = 0.999 (보통)
추론 시 ema_weight 사용
```

### 2-5. Checkpoint Soup (BirdCLEF 2위 전략)

**개념:** 같은 모델의 여러 epoch 가중치를 평균 → 단일 best보다 안정적

```python
# epoch 13~50 중 CV 개선된 것들의 가중치 평균
soup_weight = mean([ckpt_ep13, ckpt_ep20, ckpt_ep35, ...])
```

early stopping 대신 이걸 쓰면 더 안정적인 경우가 많다.

---

## 3. Preprocessing / Feature Extraction

### 3-1. 오디오 전처리 표준

```python
# 상위권 표준 전처리 순서
1. load_audio(path)              # librosa/soundfile
2. resample(to=target_sr)        # 32000Hz 표준 (BirdCLEF 기준)
3. to_mono()                     # stereo → mono
4. normalize(std=1.0)            # 입력 정규화 (std=1로 맞춤)
5. fix_length(5.0 sec)           # crop or pad
```

**왜 32000Hz인가:** 새소리, 환경음 등 대부분의 음향 신호는 16kHz 이하 주파수 대역에 집중.
나이퀴스트 정리에 의해 최대 주파수의 2배 이상 SR이면 충분. 32kHz면 16kHz까지 커버.

### 3-2. Feature 종류와 선택

**개념: 각 feature가 뭘 보는가**

| Feature | x축 | y축 | 강점 | 약점 |
|---|---|---|---|---|
| Mel-spectrogram | 시간 | 멜 주파수 | 범용, 인간 청각에 근사 | 위상 정보 없음 |
| MFCC | 시간 | 켑스트럼 계수 | 음성 특화, 압축됨 | 스펙트럼 세부 손실 |
| CQT | 시간 | 로그 주파수 | 악기음, 하모닉 | 계산 느림 |
| Raw waveform | 시간 | 진폭 | 정보 손실 없음 | 모델이 더 복잡해야 함 |

**상위권 대부분의 선택: Log-Mel Spectrogram**
- 범용성이 가장 높고 ImageNet pretrained 모델과 궁합이 좋음
- 처음엔 무조건 mel부터 시작

**파라미터 표준값:**
```yaml
sample_rate: 32000
n_fft: 1024
hop_length: 320      # = 10ms stride (32000/320 = 100 frame/sec)
n_mels: 128
fmin: 50
fmax: 16000
power: 2.0           # power spectrogram
to_db: true          # dB 스케일로 변환
image_size: [224, 224]  # EfficientNet 기본
```

### 3-3. 3채널 구성 전략

**왜 3채널인가:** ImageNet pretrained 모델은 RGB 3채널 입력을 기대함.
1채널 mel을 3채널로 만드는 방법이 몇 가지 있다.

| 방식 | 구성 | 특징 |
|---|---|---|
| `repeat` | [mel, mel, mel] | 가장 단순. 빠른 시작에 좋음 |
| `delta` | [mel, Δmel, ΔΔmel] | 시간 변화율 추가. 음향 표준 |
| `multi_res` | [mel_512, mel_1024, mel_2048] | 다해상도. 정보 풍부 |

**delta 개념:**
```
Δmel  = mel의 1차 미분 (시간에 따른 변화율)
ΔΔmel = mel의 2차 미분 (변화율의 변화율)
정적 스펙트럼 + 동적 변화를 동시에 표현 → 음향 인식에서 검증된 기법
```

---

## 4. Augmentation 전략

### 개념: 왜 증강이 중요한가

오디오 데이터는 종종:
- 녹음 환경이 다양 (실내/실외, 마이크 종류)
- 배경 노이즈가 섞임
- 같은 소리도 거리·방향에 따라 달라짐

증강 = 이 다양성을 인위적으로 만들어서 모델이 robust해지도록.

### 4-1. Waveform 증강 (CPU, dataset 안에서)

```python
# 상위권이 공통으로 쓰는 것들
TimeShift    # 오디오를 랜덤하게 앞뒤로 이동 (circular)
             # 가장 간단하고 효과적. BirdCLEF 3위도 핵심 증강으로 사용
GainAugment  # 볼륨을 0.5~1.5배 랜덤 조절
AddNoise     # 가우시안 노이즈 추가 (SNR 20~40dB)
PitchShift   # 음높이 변경 (±2 semitone)
             # 계산 비용이 크므로 p=0.3 정도로 적게
```

### 4-2. Spectrogram 증강 (GPU, trainer 안에서)

**SpecAugment — 음향 증강의 표준**

```
개념: spectrogram에서 주파수 or 시간 구간을 마스킹
     마스킹된 부분 없이도 분류하도록 강제 → robustness↑

FreqMask: 연속된 주파수 대역 n개를 0으로 (n_mels의 20% 이내)
TimeMask: 연속된 시간 프레임 n개를 0으로 (전체의 20% 이내)
보통 2개씩 적용
```

```python
# 파라미터 기준값
freq_mask_param = 24   # n_mels=128의 경우
time_mask_param = 40   # image_width의 경우
n_freq_masks = 2
n_time_masks = 2
```

### 4-3. Mixup — 상위권 필수

**개념:**
```
두 샘플을 섞는다:
x_mixed = λ * x1 + (1-λ) * x2
y_mixed = max(y1, y2)  ← multilabel에서는 max (둘 다 있다고 간주)

λ는 Beta(α, α) 분포에서 샘플링 (α=0.5가 표준)

왜 효과적인가:
- 모델이 명확한 경계 대신 soft decision을 학습
- 과적합 방지
- 특히 클래스 불균형에서 minority class 노출 빈도↑
```

**주의:** multilabel에서 label을 average하면 틀림. `max`를 써야 한다.
(두 오디오가 섞이면 둘 다 들림 → 둘 다 1)

### 4-4. 증강 강도 가이드

```
데이터 많음  → 증강 약하게 (이미 다양함)
데이터 적음  → 증강 강하게 (다양성 인위 생성 필요)
노이즈 많음  → AddNoise 약하게 or 끔
도메인 특수  → 도메인 특화 증강 우선 (예: 배경음 추가)
```

---

## 5. Error Analysis

### 5-1. Confusion Matrix 분석

```python
# 구현할 것들
1. per-class F1 / Recall / Precision 테이블
2. confusion matrix (상위 K개 오분류 클래스)
3. 가장 많이 틀리는 클래스 top 10
```

**패턴별 해석:**
```
A → B로 자주 틀림: A와 B가 음향적으로 유사. 더 세밀한 feature 필요
특정 클래스 recall=0: 학습 샘플이 너무 적음 → oversampling or class_weight
precision 낮음: 다른 소리를 이 클래스로 오분류 → threshold 높이기
```

### 5-2. 신뢰도 기반 분석

```python
# 잘못 분류된 샘플의 confidence 분포
wrong_high_conf  # 높은 확률로 틀림 → 라벨 오류 가능성 or 매우 어려운 샘플
wrong_low_conf   # 낮은 확률로 틀림 → 모델이 불확실 → 더 학습 필요

# 확인 방법
val_probs[wrong_indices].max(axis=1).hist()  # 오분류 샘플의 max probability 분포
```

### 5-3. 오디오 직접 듣기

상위권들이 반드시 하는 것. 수치만 보면 놓치는 것들:
```
□ 자주 틀리는 클래스 5개 → 실제 오디오 10개씩 청취
□ label 오류 찾기 (경쟁에서 노이즈 라벨은 흔함)
□ 어떤 상황(배경 노이즈, 거리 등)에서 틀리는지 파악
→ 이걸 알면 어떤 증강을 추가할지 결정할 수 있음
```

### 5-4. OOF (Out-of-Fold) 분석

```
OOF = K-fold에서 각 fold의 validation 예측을 모은 것
    = 전체 train 데이터에 대한 예측

OOF 분석:
- OOF score vs LB score 차이 → 일정하면 CV 신뢰 가능
- 어떤 fold에서 많이 틀리는지 → 데이터 분포 문제 찾기
- OOF로 threshold 최적화 (test에 leakage 없이)
```

---

## 6. Post-Processing 스킬

### 6-1. Threshold 최적화

**개념:** 모델은 0~1 사이 확률을 출력한다.
0.5로 자르면 최적이 아닐 수 있다. 특히 불균형 데이터에서.

```python
# Global threshold
best_thr = argmax over [0.1, 0.2, ..., 0.9]:
    metric(y_true, (y_prob > thr).astype(int))

# Per-class threshold (더 강력)
for c in range(num_classes):
    best_thr[c] = argmax: metric_c(y_true[:, c], y_prob[:, c] > thr)

# OOF로 탐색해야 test leakage 없음
```

**iterative refinement (DCASE 상위권 방법):**
```
1. 각 클래스 threshold를 class-wise F1 최대화로 초기화
2. 랜덤 클래스 선택 → 그 클래스의 threshold를 micro-F1 최대화로 재조정
3. 개선 없을 때까지 반복
→ global F1이 local 최적보다 높아짐
```

### 6-2. Temporal Smoothing

**개념:** 긴 오디오(soundscape)를 sliding window로 추론하면
인접한 window 예측이 일관성이 없을 수 있다.
시간축으로 smoothing을 걸면 안정화된다.

```python
# BirdCLEF 3위 사용: 가중 평균 smoothing
weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # 중앙 가중치 최대
smoothed = np.convolve(predictions, weights, mode='same')
```

### 6-3. Label Smoothing

**개념:** hard label(0 or 1) 대신 soft label(0.05 or 0.95)로 학습.
과적합 방지. 특히 라벨 노이즈가 있을 때 효과적.

```yaml
loss.label_smoothing: 0.1  # 0이 0.05로, 1이 0.95로
```

### 6-4. Test Time Augmentation (TTA)

**개념:** 같은 테스트 샘플을 여러 방식으로 변환해서 각각 추론 → 평균.
단일 추론보다 더 안정적인 예측.

```python
# 음향에서 효과적인 TTA
original           # 원본
time_shift_0.5s   # 0.5초 shift
time_shift_-0.5s  # -0.5초 shift
speed_0.9x        # 속도 0.9배 (피치 변화 없이)
speed_1.1x        # 속도 1.1배

# 전부 평균
final_pred = mean([pred_orig, pred_shift1, pred_shift2, ...])
```

### 6-5. Ensemble

**상위권의 앙상블 전략:**

```
다양성이 핵심. 같은 모델 5개 평균보다 다른 모델 3개 평균이 낫다.

다양성 확보 방법:
1. 다른 backbone (b0, b4, nfnet, convnext)
2. 다른 feature (mel, delta, multi_res)
3. 다른 mel 파라미터 (n_mels, image_size)
4. 다른 fold
5. 다른 augmentation 강도

앙상블 방법:
- 단순 평균 (가장 기본)
- 가중 평균 (fold val_score로 가중치)
- rank averaging (score를 rank로 변환 후 평균 → outlier에 강건)
```

### 6-6. Pseudo Labeling

**개념:** unlabeled 데이터(또는 test 데이터)에 모델이 예측한 label을 붙여서
다시 학습 데이터로 사용. 데이터가 늘어나는 효과.

```
BirdCLEF 1~3위 전부 사용.

단계:
1. 1차 모델로 unlabeled 추론
2. 확률 > 0.7인 예측만 hard label 채택 (신뢰도 필터)
3. pseudo 데이터 + original 데이터로 2차 학습
4. 2차 모델로 다시 pseudo label 갱신 (반복 가능)

주의: 
- pseudo label threshold가 너무 낮으면 오염됨
- 처음엔 0.7~0.8로 보수적으로 시작
- batch에서 pseudo:original 비율을 1:1 정도로 유지
```

---

## 7. 무조건 지켜야 할 엄격한 규칙

### 7-1. 시드 고정 (재현성)

```python
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 재현성 vs 속도 tradeoff
```

**왜:** 같은 코드가 다른 결과를 내면 실험 비교 자체가 불가능.
seed가 다른 것과 실제 개선을 구분할 수 없다.

### 7-2. Config 저장 (재현성)

```python
# 매 실험 시작 시
save_config(cfg, output_dir / 'config.yaml')
# 나중에 이 config를 로드하면 완전히 동일한 실험 재현 가능
```

### 7-3. OOF 저장

```python
# 매 fold 검증 후
np.save(f'outputs/{exp}/oof_fold{fold}.npy', val_preds)
np.save(f'outputs/{exp}/oof_labels.npy', val_labels)
# threshold 최적화, 앙상블 가중치 계산, error analysis에 사용
```

### 7-4. CV-LB 기록 테이블

```
| exp_name      | CV    | LB    | 비고              |
|---------------|-------|-------|-------------------|
| exp001_base   | 0.721 | 0.718 | baseline          |
| exp002_mixup  | 0.734 | 0.731 | mixup alpha=0.5   |
| exp003_nfnet  | 0.741 | 0.739 | backbone 교체     |

CV와 LB의 차이가 일정한지 확인.
CV가 오르면 LB도 오를 거라는 신뢰 확보.
```

### 7-5. Data Leakage 방지

```
가장 흔한 실수:

□ fold 분할 전에 전처리를 전체 데이터로 하면 leakage
  → 반드시 fold 분할 후 train으로만 통계 계산

□ test 데이터 정보가 train에 섞이면 leakage
  → threshold 최적화는 반드시 OOF로만

□ 같은 사람/기기의 녹음이 train/valid에 섞이면 leakage
  → GroupKFold 사용
```

### 7-6. 제출 파일 검증

```python
# 제출 전 반드시 확인
assert len(submission) == len(test_df)  # 행 수
assert submission.isnull().sum().sum() == 0  # NaN 없음
assert (submission[prob_cols] >= 0).all().all()  # 음수 없음
assert (submission[prob_cols] <= 1).all().all()  # 1 초과 없음
print(submission.head())  # 눈으로 확인
```

---

## 8. 대회 중 추가로 대비해야 할 것들

### 8-1. CV-LB 불일치 대응

```
증상: CV는 오르는데 LB가 안 오름 (또는 반대)

원인:
1. fold 전략이 test 분포와 다름
2. train/test 도메인 시프트
3. 평가지표 구현 오류

대응:
1. fold 전략 재검토 (group 있는지)
2. test 데이터 분포 EDA
3. metrics.py와 대회 공식 지표 비교 검증
```

### 8-2. GPU OOM 대응

```
즉각 해결:
- batch_size 반으로 → grad_accum 2배 (effective batch size 유지)
- image_size 줄이기 (256→224→192)
- fp16 AMP 확인 (켜져 있는지)

근본 해결:
- 더 작은 backbone으로 교체
- gradient checkpointing 활성화
```

### 8-3. 학습이 수렴 안 할 때

```
체크리스트:
□ lr이 너무 큼 → 10배 줄이기
□ label이 logit에 들어가야 하는데 sigmoid 통과값이 들어감
  (BCEWithLogitsLoss는 logit을 받음)
□ normalization이 안 됨
□ batch 안에 NaN 있음 (log(0) 등)
□ weight decay가 너무 큼
```

### 8-4. 시간 관리

```
무박 2일에서 가장 큰 실수: 하나에 너무 오래 매달림

규칙:
- 실험 하나에 최대 2시간 투자. 안 되면 다음으로.
- 1-fold로 빠르게 검증 후 좋으면 5-fold
- 마지막 6시간은 새 실험 금지 → 앙상블·제출 선택에 집중
- 제출은 마감 1시간 전에 완료 (서버 문제 대비)
```

---

## 9. 상위권의 초반 움직임 요약

```
Hour 0~1:
  → 데이터 받자마자 EDA (클래스 분포, 오디오 청취)
  → CV 전략 확정 (어떤 fold를 쓸지)
  → 파이프라인 e2e 통과 확인

Hour 1~2:
  → 1-fold baseline 제출 (점수보다 파이프라인 확인)
  → CV-LB 상관 확인
  → discussion 탭 확인 (다른 사람 EDA 노트북 참고)

Hour 2~6:
  → feature 실험 (mel 파라미터)
  → backbone 1~2개 비교
  → 결과를 표로 기록

Hour 6~12:
  → 가장 좋은 설정으로 5-fold 학습
  → mixup + specaugment 추가
  → OOF 분석으로 약한 클래스 파악
```

---

## 💡 한 줄 핵심 요약

1. **CV가 전부다** — CV 신뢰 없이는 아무 실험도 의미 없다
2. **mel+EfficientNet이 기본** — 다른 거 다 실패해도 이건 작동한다
3. **mixup은 거의 항상 도움** — 음향 대회에서 거의 필수
4. **앙상블의 핵심은 다양성** — 같은 모델 여러 개보다 다른 모델 몇 개
5. **OOF를 항상 저장** — threshold 최적화, error analysis, 앙상블 전부 여기서 나옴
6. **마지막 6시간은 새 실험 금지** — 마무리와 제출 선택에 집중
