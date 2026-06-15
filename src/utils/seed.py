# Global seed fixation for reproducibility.
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, benchmark: bool = True) -> None:
    """Fix random / numpy / torch / cudnn seeds.

    benchmark=True: cuDNN이 입력 크기 별로 최적 알고리즘 선택 → 5~15% speedup.
                    대신 deterministic 보장 해제 (Kaggle에서는 실질적 영향 없음).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cudnn.benchmark = benchmark
    # TF32: A100에서 matmul ~3×, conv ~1.5× speedup. 정밀도 10-bit mantissa → 음향분류 충분
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
