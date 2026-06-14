# IMPLEMENTATION.md

> Claude Code가 각 파일을 **무엇을, 어떤 인터페이스로** 구현해야 하는지 적은 작업 지시서.
> `CLAUDE.md`의 불변 규칙을 전제로 한다. 모든 모듈은 아래 함수 시그니처를 따른다.
> 시그니처가 정해져 있으면 모듈끼리 서로를 안전하게 호출할 수 있다.

---

## 0. Config 스키마 (이게 전체의 뼈대)

모든 코드는 이 config를 읽는다. `default.yaml`이 베이스이고 실험 yaml이 일부만 override 한다.

```yaml
# ===== configs/default.yaml =====
exp_name: default
seed: 42
debug: false                  # true면 데이터 1% + epoch 1 + wandb disabled

data:
  train_csv: input/train_metadata.csv
  audio_dir: input/train_audio
  folds_csv: input/folds.csv
  label_col: primary_label     # 메타데이터의 라벨 컬럼명
  id_col: filename
  group_col: null              # GroupKFold 쓸 경우 그룹 컬럼 (예: author/recording_id)
  multilabel: true
  sample_rate: 32000
  duration: 5.0                # 클립 길이(초)
  crop: random                 # random | first | center  (학습 시 crop 방식)

feature:
  type: melspec                # melspec | mfcc | cqt | raw   ← 갈아끼우는 지점
  compute_on: gpu              # gpu(frontend 레이어) | cpu(cache_features 사용)
  n_fft: 1024
  hop_length: 320
  win_length: 1024
  n_mels: 128
  fmin: 50
  fmax: 16000
  power: 2.0
  to_db: true
  normalize: true              # per-image 정규화
  image_size: [256, 256]       # backbone 입력 크기 (null이면 리사이즈 안 함)
  n_channels: 3                # 1 | 3.  3채널 만드는 방식은 channel_mode로
  channel_mode: repeat         # repeat | delta | multi_res (아래 설명)

model:
  type: timm                   # timm | panns | wav2vec2   ← 갈아끼우는 지점
  backbone: tf_efficientnet_b0_ns
  pretrained: true
  in_chans: 3
  num_classes: 182             # prepare_folds가 자동 채울 수도 있음
  head: linear                 # linear | sed | attention
  drop_rate: 0.2
  drop_path_rate: 0.2

train:
  folds: [0]                   # 이번 실행에서 돌릴 fold 목록
  n_folds: 5
  epochs: 10
  batch_size: 32
  num_workers: 4
  amp: true
  grad_accum: 1
  clip_grad: null
  ema: false

augment:
  mixup: 0.0                   # >0이면 mixup alpha
  mixup_mode: max              # max(다중레이블) | mean
  spec_augment: false
  freq_mask: 24
  time_mask: 40
  gain: false                  # waveform 게인 증강
  noise: false                 # waveform 노이즈 추가
  pitch_shift: false
  time_shift: false

loss:
  type: bce                    # bce | focal | ce | lsep
  focal_gamma: 2.0
  label_smoothing: 0.0
  class_weights: null

optimizer:
  type: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: cosine            # cosine | onecycle | plateau | none
  warmup_ratio: 0.1
  min_lr: 1.0e-6

metric:
  name: f1                     # f1 | macro_f1 | auc | cmap   ← 대회 공식 지표로 교체
  monitor: val_f1              # checkpoint 기준 지표
  mode: max                    # max | min

inference:
  tta: [none]                  # none | time_shift | sliding_window
  sliding_window: false        # 긴 오디오를 겹쳐 잘라 평균
  window_stride: 2.5
  ensemble_ckpts: []           # 앙상블할 가중치 경로 목록

wandb:
  project: kaggle-audio-clf
  entity: null
  mode: online                 # online | offline | disabled
  tags: []
```

> **channel_mode 설명** (1D→3채널 변환 전략, frontend가 처리):
> - `repeat`: 같은 mel을 3채널 복제 (가장 단순, ImageNet pretrained와 궁합 좋음)
> - `delta`: [mel, delta, delta-delta] 스택 (음향 표준 기법)
> - `multi_res`: 서로 다른 n_fft/hop로 만든 3개 스펙트로그램 스택 (사용자가 말한 "3채널 병합")

---

## 1. utils/

### `utils/config.py`
```python
def load_config(path: str, overrides: list[str] | None = None) -> OmegaConf:
    """default.yaml을 로드 후 path의 실험 yaml로 deep-merge,
    마지막에 CLI overrides("train.epochs=2") 적용해 반환."""

def save_config(cfg, out_dir: Path) -> None:
    """실행 시점 config를 outputs/<exp>/config.yaml로 덤프 (재현성)."""
```
- OmegaConf 사용. `debug=true`면 여기서 epochs=1, data 샘플링, wandb.mode=disabled 강제.

### `utils/seed.py`
```python
def seed_everything(seed: int) -> None:
    """random / numpy / torch / cudnn 시드 고정."""
```

### `utils/logger.py`
```python
class Logger:
    """wandb 래퍼. cfg.wandb.mode=='disabled'면 모든 메서드가 no-op."""
    def __init__(self, cfg): ...
    def log(self, metrics: dict, step: int | None = None) -> None: ...
    def log_audio(self, name, waveform, sr) -> None: ...   # 디버그용
    def watch(self, model) -> None: ...
    def finish(self) -> None: ...
```
- **wandb가 없거나 disabled여도 학습이 멈추지 않아야 한다.** 모든 호출을 안전하게 감싼다.

### `utils/metrics.py`
```python
def get_metric_fn(cfg) -> Callable:
    """cfg.metric.name에 맞는 (y_true, y_pred)->float 함수 반환.
    f1/macro_f1/auc/cmap 등. 대회 공식 지표를 여기에 구현."""
```
- 다중레이블이면 threshold 적용 후 계산. cmAP(class-mean average precision)는 음향 대회 단골이니 포함.

### `utils/checkpoint.py`
```python
def save_checkpoint(model, optimizer, epoch, score, path: Path) -> None: ...
def load_checkpoint(path: Path, model, optimizer=None) -> dict: ...

class BestTracker:
    """cfg.metric.mode 기준 best score 추적, best일 때만 저장."""
    def update(self, score, model, optimizer, epoch, path) -> bool: ...
```

---

## 2. data/  (전부 CPU. waveform 텐서까지만 만든다)

### `data/preprocessing.py`  [신규]
```python
def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """librosa/soundfile로 로드 + target_sr로 리샘플 + 모노 변환."""

def trim_silence(wav: np.ndarray, top_db: float = 30) -> np.ndarray: ...
def normalize_wave(wav: np.ndarray) -> np.ndarray:        # peak/RMS 정규화
    ...
def fix_length(wav: np.ndarray, length: int, mode: str) -> np.ndarray:
    """길이 < target이면 pad/반복, 길면 crop(random/first/center)."""
```
- 순수 함수로. dataset.py가 이걸 조합해서 쓴다.

### `data/dataset.py`
```python
class AudioDataset(Dataset):
    """메타데이터 CSV의 한 행 → (waveform_tensor, label_tensor) 반환.
    feature 계산은 여기서 하지 않는다(그건 frontend/GPU 몫).
    단, cfg.feature.compute_on=='cpu'면 캐시된 .npy feature를 로드."""
    def __init__(self, df, cfg, mode='train'): ...
    def __getitem__(self, i):
        # 1) preprocessing.load_audio
        # 2) fix_length (+ crop)
        # 3) (train) waveform 증강 = augment.apply_waveform_aug
        # 4) label → multi-hot or index
        # return wav (1, T) , label
```
- `mode`: `train`/`valid`/`test`. test는 라벨 없이 id 반환.
- 라벨 인코딩(클래스명→인덱스) 매핑은 `prepare_folds.py`가 만든 `label_map.json`을 읽는다.

### `data/dataloader.py`
```python
def build_dataloader(df, cfg, mode) -> DataLoader:
    """AudioDataset + DataLoader. shuffle/drop_last는 mode로 결정.
    num_workers, pin_memory, persistent_workers 세팅."""
```

### `data/augment.py`
```python
def apply_waveform_aug(wav: np.ndarray, cfg) -> np.ndarray:
    """gain / noise / pitch_shift / time_shift (config 토글)."""

class SpecAugment(nn.Module):
    """freq mask + time mask. frontend 뒤 GPU에서 적용."""
    def forward(self, spec): ...

def mixup_batch(x, y, alpha, mode='max'):
    """배치 단위 mixup. mode='max'면 라벨은 두 라벨의 max(다중레이블).
    trainer.py가 학습 스텝에서 호출."""
    return x_mixed, y_mixed
```
- waveform 증강은 CPU(dataset 안), spectrogram 증강(SpecAugment)·mixup은 GPU(trainer 안). 위치를 헷갈리지 마라.

### `data/feature_extractor.py`  [신규, CPU 캐싱 경로]
```python
def extract_feature(wav: np.ndarray, cfg) -> np.ndarray:
    """cfg.feature.type에 따라 mel/mfcc/cqt 계산 → (C,H,W) np.float32.
    frontend.py의 GPU 버전과 동일한 출력 형태를 보장해야 한다."""
```
- **frontend.py(GPU)와 출력 규격이 같아야** 두 경로를 자유롭게 바꿔 쓸 수 있다. 이 일관성이 핵심.

---

## 3. models/  (전부 GPU. waveform → logits)

### `models/frontend.py`  [핵심]
```python
class Frontend(nn.Module):
    """waveform (B,1,T) → image (B,C,H,W). GPU에서 동작.
    cfg.feature.type으로 변환 종류 분기. nnAudio/torchaudio 사용."""
    def __init__(self, cfg): ...
    def forward(self, wav):
        # 1) STFT/Mel/MFCC/CQT (type 분기)
        # 2) to_db, normalize
        # 3) channel_mode(repeat/delta/multi_res)로 C채널 구성
        # 4) image_size로 resize
        # return (B, C, H, W)
```
- 변환 레이어는 가능하면 미분 가능/배치 가능한 것으로(nnAudio.Spectrogram, torchaudio.transforms).
- **새 feature 추가 = 여기 분기 한 개 + feature_extractor에 대응 함수 한 개.** 다른 곳은 안 건드린다.

### `models/backbones.py`
```python
def build_backbone(cfg) -> tuple[nn.Module, int]:
    """timm.create_model(cfg.model.backbone, pretrained, in_chans, num_classes=0,
    features_only=False)로 backbone 생성. (backbone, feature_dim) 반환."""
```

### `models/pretrained_audio.py`  [신규]
```python
def build_audio_pretrained(cfg) -> tuple[nn.Module, int]:
    """PANNs(CNN14) / wav2vec2 / BirdNET 등 음향 사전학습 모델 로드.
    이들은 waveform을 직접 받으므로 frontend 없이 쓰는 경로일 수 있음.
    backbones.build_backbone과 같은 (module, feat_dim) 규격으로 반환."""
```
- 대회 후반 고도화용. 처음엔 빈 스텁 + NotImplementedError 메시지로 둬도 된다.

### `models/heads.py`
```python
class LinearHead(nn.Module): ...        # 평범한 분류
class AttentionHead(nn.Module): ...     # attention pooling (PANNs식)
class SEDHead(nn.Module):
    """clip-level + frame-level 동시 출력 (Sound Event Detection)."""
def build_head(cfg, feat_dim) -> nn.Module: ...
```

### `models/factory.py`  [신규, 조립 담당]
```python
class AudioModel(nn.Module):
    """frontend + backbone + head를 묶은 최종 모델.
    forward(wav) -> logits.  cfg.feature.compute_on=='cpu'면 frontend 생략."""
    def __init__(self, cfg): ...
    def forward(self, x):
        # x가 waveform이면 frontend 통과, 이미 feature면 바로 backbone
        ...

def build_model(cfg) -> AudioModel:
    """cfg.model.type 분기: timm → backbones, panns/wav2vec2 → pretrained_audio.
    여기가 model을 갈아끼우는 단일 지점."""
```

---

## 4. training/

### `training/losses.py`
```python
def build_loss(cfg) -> nn.Module:
    """bce(BCEWithLogitsLoss) | focal | ce | lsep 분기.
    class_weights, label_smoothing 반영."""
```

### `training/optimizers.py`
```python
def build_optimizer(model, cfg) -> Optimizer: ...
def build_scheduler(optimizer, cfg, steps_per_epoch) -> Scheduler:
    """cosine/onecycle/plateau/none + warmup_ratio."""
```

### `training/trainer.py`
```python
class Trainer:
    """한 fold의 학습 전체를 책임진다. K-Fold 루프는 main.py에."""
    def __init__(self, model, loaders, cfg, logger, fold): ...
    def train_one_epoch(self, epoch) -> dict:
        # AMP(autocast+GradScaler), mixup_batch, SpecAugment, grad_accum, clip_grad
    def validate(self) -> dict:
        # no_grad, metric 계산, oof 예측 저장
    def fit(self) -> float:
        # epoch 루프 + BestTracker + logger.log + 스케줄러 step
        # return best_score
```
- mixup/specaug는 여기 `train_one_epoch`의 GPU 스텝 안에서 적용.
- validation 예측은 oof(out-of-fold)로 모아 `outputs/<exp>/oof_fold{n}.npy`에 저장.

---

## 5. postprocess/

### `postprocess/threshold.py`
```python
def optimize_threshold_global(y_true, y_pred, metric='f1', n_steps=100) -> float:
    """전역 단일 threshold를 grid search로 최적화. y_pred는 sigmoid 확률."""

def optimize_threshold_per_class(y_true, y_pred, n_steps=100) -> np.ndarray:
    """클래스별 threshold 독립 최적화. 반환: (num_classes,) array."""

def apply_threshold(y_pred, thresholds: float | np.ndarray) -> np.ndarray:
    """확률 예측 (N, C)에 threshold 적용 → binary (N, C)."""

def smooth_predictions(y_pred, window=3) -> np.ndarray:
    """시계열 예측에 이동 평균 스무딩 (SED / sliding window 용)."""

def load_oof_and_optimize(oof_paths, label_path, mode='global', n_steps=100):
    """여러 fold OOF .npy를 합쳐 최적 threshold 계산.
    mode: 'global' | 'per_class'"""
```

---

## 6. scripts/

### `scripts/prepare_folds.py`  [신규]
- 메타데이터 로드 → 라벨맵(`label_map.json`) 생성 → `cfg.data.group_col` 있으면 GroupKFold,
  없으면 StratifiedKFold(다중레이블이면 iterative-stratification) → `fold` 컬럼 붙여 `folds.csv` 저장.
- `num_classes`를 출력해서 config에 반영하도록 안내 메시지 출력.

### `scripts/verify_audio.py`  [신규]
- 전체 오디오 순회하며: 로드 실패(손상), sample_rate 불일치, 길이 0/무음, NaN 검출 → 리포트 CSV.
- `try/except`로 개별 파일 에러를 모아서 보고(여기서만 예외 삼키기 허용).

### `scripts/cache_features.py`
- `feature_extractor.extract_feature`로 전 파일 feature 계산 → `outputs/<exp_name>/cache/<stem>.npy` 저장.
- `--workers` 인자로 멀티프로세싱(ProcessPoolExecutor). 이미 있으면 skip(resume 가능).
- `image_size` 리사이즈는 PIL로 적용 후 저장.

### `scripts/train_all_folds.sh`
```bash
#!/bin/bash
# usage: bash scripts/train_all_folds.sh configs/exp002_advanced.yaml
CFG=$1
for f in 0 1 2 3 4; do
  python main.py --config $CFG train.folds=[$f]
done
```

### `scripts/make_submission.sh`
- `inference.py` 호출 → `submission.csv` 생성 → (선택) kaggle CLI 제출 명령 주석으로.

---

## 7. main.py  (K-Fold 오케스트레이션만)

```python
def main():
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    logger = Logger(cfg)
    df = pd.read_csv(cfg.data.folds_csv)

    scores = []
    for fold in cfg.train.folds:
        train_df = df[df.fold != fold]; valid_df = df[df.fold == fold]
        loaders = {m: build_dataloader(d, cfg, m)
                   for m, d in [('train', train_df), ('valid', valid_df)]}
        model = build_model(cfg).cuda()
        trainer = Trainer(model, loaders, cfg, logger, fold)
        scores.append(trainer.fit())
    logger.log({'cv_mean': np.mean(scores)})
    logger.finish()
```
- **학습 루프 본문을 여기 쓰지 마라.** 여기는 fold를 돌리고 Trainer를 부르는 일만.

---

## 8. inference.py  (추론 + TTA + 앙상블)

```bash
# 사용법
python inference.py --config <yaml> --ckpt <dir_or_file> [--threshold 0.5] [--out submission.csv]
```

```python
def main():
    # --ckpt: 단일 .pth 또는 best_fold*.pth 가 있는 디렉토리
    ckpt_paths = _find_checkpoints(args.ckpt)
    # 각 체크포인트 모델 로드 → predict_one_model (TTA 적용)
    # TTA 모드: cfg.inference.tta (none/flip/gain_up/gain_down)
    # 모델 앙상블: np.mean(ensemble_preds)
    # sigmoid 확률 기준 threshold 적용 → label 텍스트 매핑
    # submission.csv 저장: id_col + 'prediction' (공백 구분 레이블)
```

- `label_map.json` (prepare_folds가 생성)으로 idx→레이블 역매핑
- 앙상블 가중치 지원 예정; 현재는 단순 평균

---

## 9. 현재 구현 상태

**CLAUDE.md §7 구현 순서 기준 전 단계 완료.**

| 단계 | 파일 | 상태 |
|------|------|------|
| 1 | utils/config, seed, logger | ✅ |
| 2 | configs/default.yaml, exp001_baseline.yaml | ✅ |
| 3 | data/dataset, dataloader | ✅ |
| 4 | models/frontend, backbones, heads, factory | ✅ |
| 5 | training/losses, optimizers, trainer | ✅ |
| 6 | utils/metrics, checkpoint | ✅ |
| 7 | main.py (1-fold e2e, dummy data 통과) | ✅ |
| 8 | scripts/prepare_folds, verify_audio | ✅ |
| 9 | data/augment, preprocessing | ✅ |
| 10 | inference.py + TTA | ✅ |
| 11 | data/feature_extractor, scripts/cache_features | ✅ |
| 12 | models/pretrained_audio, postprocess/threshold | ✅ |

새 기능 추가 시 이 표에도 반영할 것.
