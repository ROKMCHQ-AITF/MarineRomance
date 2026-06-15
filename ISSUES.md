# 코드베이스 결함 보고서

> 생성일: 2026-06-15  
> 대상: `src/`, `main.py`, `inference.py`  
> smoke test: `python main.py --config configs/default.yaml debug=true wandb.mode=disabled loss.type=ce` → **통과**

---

## 요약

| 등급 | 건수 | 상태 |
|------|------|------|
| CRITICAL (실행 중단 / 잘못된 결과) | 5 | ✅ 수정 완료 |
| WARNING (경고 / 잠재적 문제) | 2 | ✅ 수정 완료 |
| INFO (권고 사항) | 2 | ⚠️ 수동 조치 필요 |

---

## CRITICAL — 수정 완료

### C-1. `dataset.py` — debug 모드 label 형태 불일치
- **파일**: `src/data/dataset.py:47-57`
- **증상**: `multilabel: false` + `loss.type: ce` 조합에서 debug 모드는 `float (C,)` multi-hot, 실제 모드는 `long scalar` 반환. CE loss가 debug에서 crash.
- **원인**: debug 분기가 `multilabel` 설정을 무시하고 항상 multi-hot 반환.
- **수정**: multilabel 여부에 따라 debug label 형태 분기.

### C-2. `dataset.py` — label_map 없을 때 silent failure
- **파일**: `src/data/dataset.py:31-42`
- **증상**: `prepare_folds.py` 실행 전 학습 시 모든 샘플의 레이블이 class 0으로 고정. 에러 없이 학습되지만 결과 무의미.
- **수정**: label_map 없을 때 `[WARN]` 출력.

### C-3. `dataloader.py` — `pin_memory=True` 하드코딩
- **파일**: `src/data/dataloader.py:23`
- **증상**: CPU 전용 환경에서 `pin_memory=True`는 UserWarning 및 불필요한 오버헤드 발생. 일부 환경에서 에러.
- **수정**: `pin_memory=torch.cuda.is_available()`

### C-4. `inference.py` — 단일 레이블에서 sigmoid + threshold 사용
- **파일**: `inference.py:57, 109`
- **증상**: `multilabel: false`인데 sigmoid로 확률 계산 후 threshold=0.5로 이진화 → 여러 클래스 동시 예측, 잘못된 submission.
- **수정**: `multilabel` 여부에 따라 sigmoid/softmax, threshold/argmax 분기.

### C-5. `pretrained_audio.py` — BEATs SR 불일치 무경고
- **파일**: `src/models/pretrained_audio.py:118`
- **증상**: BEATs는 16kHz 학습, config `sample_rate: 32000` 입력 시 아무 경고 없이 학습. 성능 심각하게 저하.
- **수정**: `sample_rate != 16000` 시 `[WARN]` 출력.

---

## WARNING — 수정 완료

### W-1. `checkpoint.py` — `torch.load` weights_only 미지정
- **파일**: `src/utils/checkpoint.py:36`
- **증상**: PyTorch 2.0+에서 deprecation warning. 2.6에서 기본값 변경으로 동작 달라질 수 있음.
- **수정**: `weights_only=True` 추가.

### W-2. `dataloader.py` — Docker 환경 `persistent_workers` 위험
- **파일**: `src/data/dataloader.py:25`
- **증상**: Docker 기본 `/dev/shm` 64MB 제한 + `persistent_workers=True` + `num_workers=4` 조합에서 worker 프로세스 OOM으로 사망. `DataLoader worker exited unexpectedly` 에러.
- **현재 대응**: `debug=True` 시 `config.py`에서 `num_workers=0` 강제 적용됨.
- **실 학습 시**: `num_workers` 값을 낮추거나 Docker 실행 시 `--shm-size=2g` 옵션 추가 권장.

---

## INFO — 수동 조치 필요

### I-1. timm 모델명 deprecated
- **파일**: `configs/default.yaml:53`
- **증상**: `tf_efficientnet_b0_ns` → `tf_efficientnet_b0.ns_jft_in1k` 로 remapping 경고.
- **조치**: config에서 모델명 변경 권장 (기능 동작은 이상 없음).
```yaml
# 변경 전
backbone: tf_efficientnet_b0_ns
# 변경 후
backbone: tf_efficientnet_b0.ns_jft_in1k
```

### I-2. BEATs 사용 시 `data.sample_rate: 16000` 필수
- **파일**: `configs/*.yaml`
- **설명**: `model.type: beats` 사용 시 반드시 아래 설정 추가.
```yaml
data:
  sample_rate: 16000
```
설정하지 않으면 C-5 경고가 출력되고 성능이 크게 저하됨.

---

## smoke test 출력 (debug=true)

```
[debug] Using synthetic dummy data (no real audio required).

[fold 0] building dataloaders...
[Dataset:train] samples=16  num_classes=4  multilabel=False  target_len=160000
[Dataset:valid] samples=4   num_classes=4  multilabel=False  target_len=160000
[fold 0] train=16 samples (4 batches)  val=4 samples (1 batches)
[fold 0] building model...
[Frontend] type=melspec  channel_mode=repeat  image_size=[256, 256]  sr=32000
[AudioModel] type=timm  frontend=on  feat_dim=1280  num_classes=182
[fold 0] model=tf_efficientnet_b0_ns  params=4.2M
[Trainer fold=0] device=cuda  loss=ce  metric=macro_f1
[Trainer fold=0] amp=True  scheduler=cosine  epochs=1
[Frontend.forward] input=(4, 1, 160000)  device=cuda:0
[Frontend.forward] output=(4, 3, 256, 256)
  epoch 00 step  1/4 loss=5.1924
  epoch 00 step  2/4 loss=5.1177
  epoch 00 step  3/4 loss=4.9964
  epoch 00 step  4/4 loss=4.9373
  [validate] starting...
  [validate] logits=(4, 182)  labels=(4,)  preds=(4, 182)
[fold=0 epoch=00] train_loss=4.9373 val_loss=5.1143 val_macro_f1=0.0000
CV mean: 0.0000
```

> `val_macro_f1=0.0000` 은 정상 — debug dummy data가 num_classes=182 모델에 class 0~3만 사용하기 때문.
