# default.yaml 설정 매뉴얼

모든 실험 config는 `default.yaml`을 기반으로 값을 덮어쓴다.  
CLI override: `python main.py --config configs/exp001.yaml train.epochs=5 wandb.mode=disabled`

---

## 최상위 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `exp_name` | `default` | 실험 이름. wandb run 이름 및 outputs/ 하위 디렉토리 명에 사용됨 |
| `seed` | `42` | 전역 시드. `seed_everything()`이 학습 시작 전 가장 먼저 호출 |
| `debug` | `false` | `true`면 1배치만 돌고 종료 (파이프라인 빠른 검증용) |

---

## data

데이터 경로와 오디오 로드 설정.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `train_csv` | `input/train.csv` | 학습 메타데이터 CSV 경로 |
| `audio_dir` | `input/audio` | 오디오 파일 루트 디렉토리 |
| `folds_csv` | `input/folds.csv` | `prepare_folds.py`가 생성하는 fold 할당 CSV |
| `label_col` | `primary_label` | CSV에서 레이블 컬럼명 |
| `id_col` | `filename` | CSV에서 파일명 컬럼명 |
| `group_col` | `null` | GroupKFold용 그룹 컬럼명. null이면 StratifiedKFold |
| `multilabel` | `true` | `true`: BCEWithLogits + sigmoid. `false`: CrossEntropy + softmax |
| `sample_rate` | `32000` | 리샘플링 목표 SR (Hz) |
| `duration` | `5.0` | 학습에 사용할 클립 길이 (초) |
| `crop` | `random` | `random`: 랜덤 구간 crop / `center`: 중앙 crop / `start`: 앞에서부터 |

---

## feature

`Frontend` 모듈 (`src/models/frontend.py`) 설정. GPU에서 waveform → 이미지 텐서로 변환.

### 공통

| 키 | 기본값 | 설명 |
|----|--------|------|
| `type` | `melspec` | 특징 종류. 아래 표 참고 |
| `compute_on` | `gpu` | `gpu`: Frontend(nn.Module)로 배치 단위 변환 / `cpu`: 캐시 경로 |
| `n_fft` | `1024` | STFT FFT 크기 |
| `hop_length` | `320` | STFT hop size (샘플 수) |
| `win_length` | `1024` | STFT window 크기 |
| `n_mels` | `128` | Mel 필터 뱅크 수 |
| `fmin` | `50` | Mel 필터 최저 주파수 (Hz) |
| `fmax` | `16000` | Mel 필터 최고 주파수 (Hz). 보통 `sample_rate / 2` |
| `power` | `2.0` | STFT power. `2.0`=파워 스펙트럼, `1.0`=진폭 |
| `to_db` | `true` | AmplitudeToDB 적용 여부 (`melspec` / `lofar` / `demon` / `cqt`에 유효) |
| `normalize` | `true` | 배치 내 각 샘플을 평균 0, 분산 1로 정규화 |
| `log_eps` | `1e-6` | `log_mel`에서 log 연산 전 clamp 하한값 |
| `image_size` | `[256, 256]` | 최종 출력 이미지 크기 `[H, W]`. bilinear interpolation |
| `n_channels` | `3` | 출력 채널 수. `channel_mode`와 일치해야 함 |
| `channel_mode` | `repeat` | 3채널 구성 방법. 아래 표 참고 |

### feature.type 선택지

| 값 | 설명 | 관련 파라미터 |
|----|------|--------------|
| `melspec` | Mel 스펙트로그램 + AmplitudeToDB | `n_mels`, `fmin`, `fmax`, `power`, `to_db` |
| `log_mel` | `log(mel + ε)` — 자연로그 (dB 아님) | `log_eps` |
| `mfcc` | MFCC (이미 log-scale, `to_db` 무시) | `n_mfcc` |
| `cqt` | Constant-Q Transform (nnAudio 필요) | `n_bins`, `bins_per_octave` |
| `lofar` | 고주파수 해상도 STFT 저주파 대역 추출 | `lofar_n_fft`, `lofar_n_bins` |
| `demon` | 변조 신호 스펙트로그램 (수중 음향용) | `demon_fmin/fmax`, `demon_env_*`, `demon_n_fft` |
| `raw` | 파형 원본을 2D로 reshape | `image_size` |

### feature.channel_mode 선택지

| 값 | 채널 구성 | 설명 |
|----|----------|------|
| `repeat` | [spec, spec, spec] | 가장 단순. 단일 feature 베이스라인 |
| `delta` | [spec, Δ, ΔΔ] | 시간 미분 추가. 음성/음향에서 표준적 |
| `log_linear` | [linear, log, Δlog] | 선형/로그 스케일 동시 제공 |
| `harmonic_percussive` | [H, P, R] | HPSS로 배음/타격음/잔차 분리 |
| `multi_res` | [hop/2, hop, hop×2] | 서로 다른 시간 해상도 Mel 3장. `type=melspec` 필수 |
| `multi_feat` | [feat0, feat1, feat2] | 채널마다 다른 feature 종류. `channel_features` 리스트 지정 필요 |

#### multi_feat 전용 파라미터

```yaml
feature:
  channel_mode: multi_feat
  n_channels: 3
  channel_features:   # 채널 수만큼 지정. melspec | log_mel | lofar | demon
    - melspec
    - lofar
    - demon
```

#### HPSS 전용 파라미터

| 키 | 기본값 | 설명 |
|----|--------|------|
| `hpss_kernel_harm` | `31` | 배음 필터 커널 크기 (시간 축 median filter) |
| `hpss_kernel_perc` | `31` | 타격음 필터 커널 크기 (주파수 축 median filter) |

#### LOFAR 전용 파라미터

| 키 | 기본값 | 설명 |
|----|--------|------|
| `lofar_n_fft` | `4096` | 고해상도 FFT 크기 |
| `lofar_n_bins` | `256` | 유지할 저주파 bin 수 |

#### DEMON 전용 파라미터

| 키 | 기본값 | 설명 |
|----|--------|------|
| `demon_fmin` | `1000.0` | 대역통과 필터 하한 (Hz) |
| `demon_fmax` | `10000.0` | 대역통과 필터 상한 (Hz) |
| `demon_env_cutoff` | `50.0` | 포락선 추출용 저역통과 차단 주파수 (Hz) |
| `demon_env_sr` | `400` | 포락선 다운샘플 SR |
| `demon_n_fft` | `256` | 포락선 스펙트로그램 FFT 크기 |
| `demon_hop_length` | `8` | 포락선 스펙트로그램 hop |

---

## model

| 키 | 기본값 | 설명 |
|----|--------|------|
| `type` | `timm` | 백본 종류. `timm` / `ast` / `beats` |
| `backbone` | `tf_efficientnet_b0_ns` | `timm` 모델 ID. `timm.list_models()`로 확인 |
| `pretrained` | `true` | ImageNet pretrained weight 사용 여부 (`timm`만 해당) |
| `hf_model` | `null` | HuggingFace 모델 ID (`type=ast`일 때 사용) |
| `pretrained_path` | `input/pretrained/BEATs_iter3_plus_AS2M.pt` | BEATs 체크포인트 경로 (`type=beats`일 때 사용) |
| `in_chans` | `3` | 입력 채널 수. `feature.n_channels`와 일치해야 함 |
| `num_classes` | `182` | 출력 클래스 수. `prepare_folds.py` 실행 후 실제 값으로 변경 |
| `head` | `linear` | `linear`: 글로벌 풀링 + FC / `sed`: Sound Event Detection 헤드 |
| `drop_rate` | `0.2` | Dropout 비율 |
| `drop_path_rate` | `0.2` | Stochastic Depth 비율 |

### model.type별 동작

| 값 | Frontend 사용 | 설명 |
|----|--------------|------|
| `timm` | O | `feature` 설정대로 스펙트로그램 → timm 이미지 모델 |
| `ast` | X | HuggingFace Audio Spectrogram Transformer. 자체 전처리 내장 |
| `beats` | X | Microsoft BEATs. raw waveform 직접 입력, Frontend 건너뜀 |

---

## train

| 키 | 기본값 | 설명 |
|----|--------|------|
| `folds` | `[0]` | 학습할 fold 인덱스 목록. `[0,1,2,3,4]`면 5-fold 전체 |
| `n_folds` | `5` | 전체 fold 수 (`prepare_folds.py`와 맞춰야 함) |
| `epochs` | `10` | 에폭 수 |
| `batch_size` | `32` | 배치 크기 |
| `num_workers` | `4` | DataLoader worker 수 |
| `amp` | `true` | Automatic Mixed Precision (FP16) 사용 여부 |
| `grad_accum` | `1` | Gradient accumulation step 수. `effective_bs = batch_size × grad_accum` |
| `clip_grad` | `null` | Gradient clipping max norm. null이면 미적용 |

---

## augment

CPU 증강(`apply_waveform_aug`)과 GPU 증강(`SpecAugment`, `mixup_batch`)을 모두 포함.

| 키 | 기본값 | 적용 위치 | 설명 |
|----|--------|----------|------|
| `mixup` | `0.0` | GPU (Trainer) | Mixup alpha. `0.0`이면 비활성 |
| `mixup_mode` | `max` | GPU | `max`: 레이블 element-wise max (multi-label 안전) / `linear`: 선형 보간 |
| `spec_augment` | `false` | GPU (Trainer) | SpecAugment 활성화 여부 |
| `freq_mask` | `24` | GPU | SpecAugment 주파수 마스크 최대 크기 |
| `time_mask` | `40` | GPU | SpecAugment 시간 마스크 최대 크기 |
| `gain` | `false` | CPU (Dataset) | 랜덤 볼륨 증감 여부 |
| `gain_range` | `[0.6, 1.4]` | CPU | 볼륨 배율 범위 |
| `noise` | `false` | CPU | 가우시안 노이즈 추가 여부 |
| `noise_amp` | `0.005` | CPU | 노이즈 진폭 상한 |
| `pitch_shift` | `false` | CPU | 피치 쉬프트 여부 (librosa 필요, 느림) |
| `pitch_shift_range` | `[-2.0, 2.0]` | CPU | 피치 범위 (semitone) |
| `time_shift` | `false` | CPU | 시간축 cyclic shift 여부 |
| `time_shift_range` | `0.1` | CPU | 최대 shift 비율 (클립 길이 대비) |

---

## loss

| 키 | 기본값 | 설명 |
|----|--------|------|
| `type` | `bce` | `bce` / `focal` / `ce` / `combo` |
| `focal_gamma` | `2.0` | (`focal`) focusing parameter γ. 클수록 어려운 샘플 가중 |
| `label_smoothing` | `0.0` | `0.0`=hard label. `>0`이면 `bce`·`focal`·`ce`·`combo` 모두 적용 |

### label smoothing 공식 (multi-label)

```
y_smooth = y × (1 - s) + 0.5 × s
→ 양성(1): 1 - s/2   음성(0): s/2
```

### combo: 여러 loss 가중합

```yaml
loss:
  type: combo
  label_smoothing: 0.0        # 전체 global smoothing (component별 재정의 가능)
  components:
    - {type: focal, weight: 0.7, focal_gamma: 2.0, label_smoothing: 0.05}
    - {type: bce,   weight: 0.3}
# weight 합은 자동 정규화되므로 비율만 맞으면 됨
```

---

## optimizer

| 키 | 기본값 | 설명 |
|----|--------|------|
| `type` | `adamw` | `adamw` / `adam` / `sgd` |
| `lr` | `1e-3` | 초기 학습률 |
| `weight_decay` | `1e-4` | L2 정규화 계수 (`adamw`만 적용) |
| `scheduler` | `cosine` | `cosine` / `onecycle` / `plateau` / `none` |
| `warmup_ratio` | `0.1` | 전체 step 중 warmup 비율 (`cosine`에서 사용) |
| `min_lr` | `1e-6` | 스케줄러 하한 학습률 (`cosine`에서 `eta_min`) |

### scheduler별 동작

| 값 | 설명 |
|----|------|
| `cosine` | CosineAnnealingLR. warmup 포함, `min_lr`까지 감소 |
| `onecycle` | OneCycleLR. `lr`이 최대값, 자동 warmup/decay |
| `plateau` | ReduceLROnPlateau. val metric 기준 patience=2 |
| `none` | 스케줄러 없음. 고정 학습률 |

---

## metric

| 키 | 기본값 | 설명 |
|----|--------|------|
| `name` | `f1` | 검증 지표 종류. `f1` / `macro_f1` / `auc` / `cmap` |
| `monitor` | `val_f1` | checkpoint / early stop 기준 키 이름 |
| `mode` | `max` | `max`: 높을수록 좋음 / `min`: 낮을수록 좋음 |

### metric.name별 설명

| 값 | 설명 |
|----|------|
| `f1` | sample-average F1 (multi-label, threshold=0.5) |
| `macro_f1` | class-average F1 |
| `auc` | macro ROC-AUC |
| `cmap` | class-mean Average Precision. BirdCLEF 등에서 공식 지표 |

---

## wandb

| 키 | 기본값 | 설명 |
|----|--------|------|
| `project` | `kaggle-audio-clf` | wandb 프로젝트명 |
| `entity` | `null` | wandb 팀/유저명. null이면 기본 계정 |
| `mode` | `online` | `online` / `offline` / `disabled` |
| `tags` | `[]` | run에 붙일 태그 목록 |

> `mode: disabled`로 설정하면 모든 wandb 호출이 no-op이 됨. 디버깅 시 사용.

---

## 자주 쓰는 조합 예시

```yaml
# 빠른 파이프라인 검증
debug: true
wandb:
  mode: disabled

# 5-fold 전체 학습
train:
  folds: [0, 1, 2, 3, 4]
  epochs: 30

# BEATs + combo loss
model:
  type: beats
  pretrained_path: input/pretrained/BEATs_iter3_plus_AS2M.pt
loss:
  type: combo
  components:
    - {type: focal, weight: 0.7, focal_gamma: 2.0, label_smoothing: 0.05}
    - {type: bce,   weight: 0.3}

# 멀티 feature 채널 (melspec + lofar + demon)
feature:
  type: melspec
  channel_mode: multi_feat
  channel_features: [melspec, lofar, demon]

# SpecAugment + Mixup 풀 증강
augment:
  mixup: 0.4
  spec_augment: true
  gain: true
  noise: true
  time_shift: true
```
