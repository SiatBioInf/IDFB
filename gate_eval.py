from pathlib import Path
import sys

# Import torch before numpy/pandas on Windows to avoid c10.dll init failures.
import torch  # noqa: F401

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idfb.config import (
    GATE_MAX_LATENT_BA,
    GATE_MAX_OUTPUT_BA,
    GATE_MIN_CORR,
    GATE_MIN_CORRECTED_CORR,
    GATE_MIN_CORRECTED_VAR_RATIO,
    GATE_MIN_VAR_RATIO,
    MODEL_DIR,
    N_GPLS,
    SEED,
)
from idfb.gate import run_gate_evaluation


# Editable params
MODEL_PATH = MODEL_DIR / "vae_model.pth"
OUTPUT_DIR = MODEL_DIR / "gate"
BATCH_SIZE = 128
SEED = SEED


if __name__ == "__main__":
    print("=" * 60)
    print("IDFB post-training diagnostic (fresh probe)")
    print(f"model_path={MODEL_PATH}")
    print(f"output_dir={OUTPUT_DIR}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"seed={SEED}")
    print(f"chance≈{1.0 / N_GPLS:.3f}")
    print(
        "thresholds: "
        f"BA≤{GATE_MAX_OUTPUT_BA}/{GATE_MAX_LATENT_BA}, "
        f"corr≥{GATE_MIN_CORR}/{GATE_MIN_CORRECTED_CORR}, "
        f"var≥{GATE_MIN_VAR_RATIO}/{GATE_MIN_CORRECTED_VAR_RATIO}"
    )
    print("=" * 60)
    run_gate_evaluation(
        model_path=MODEL_PATH,
        seed=SEED,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,
    )
