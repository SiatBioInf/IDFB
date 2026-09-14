from pathlib import Path
import sys

# Import torch before numpy/pandas on Windows to avoid c10.dll init failures.
import torch  # noqa: F401

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idfb.config import (
    INPUT_DIM,
    LATENT_DIM,
    N_EPOCHS as CONFIG_N_EPOCHS,
    MODEL_DIR,
    REFERENCE_PLATFORM,
    REFERENCE_PLATFORM_ID,
    SEED,
    SELECT_BY_FIDELITY,
    SPLIT_BY_DESIGN,
    TRAIN_USE_ADVERSARIAL,
    USE_DESIGN_PAIR_LOSS,
    USE_LATENT_ADV,
    USE_REF_DECODE_LOSS,
    USE_SAMPLE_MINMAX,
    ADV_ON_CORRECTED,
    GAN_WARMUP_EPOCHS,
    LAMBDA_ADV,
    RECON_LOSS,
    DISC_LR_SCALE,
)
from idfb.gate import run_gate_evaluation
from idfb.train import run_train


BATCH_SIZE = 64
N_EPOCHS = CONFIG_N_EPOCHS
LEARNING_RATE = 2e-4
LATENT_DIM = LATENT_DIM
INPUT_DIM = INPUT_DIM
EARLY_STOPPING = True
PATIENCE = 18
SEED = SEED
USE_ADVERSARIAL = TRAIN_USE_ADVERSARIAL
SPLIT_BY_DESIGN = SPLIT_BY_DESIGN
USE_REF_DECODE = USE_REF_DECODE_LOSS
USE_DESIGN_PAIR = USE_DESIGN_PAIR_LOSS
USE_LATENT_ADV = USE_LATENT_ADV
RUN_GATE_AFTER_TRAIN = True
GATE_MODEL_PATH = MODEL_DIR / "vae_model.pth"
GATE_OUTPUT_DIR = MODEL_DIR / "gate"


if __name__ == "__main__":
    print("=" * 60)
    print("IDFB Training (VAE + GAN on corrected path)")
    print(f"use_adversarial={USE_ADVERSARIAL} adv_on_corrected={ADV_ON_CORRECTED}")
    print(f"recon_loss={RECON_LOSS} disc_lr_scale={DISC_LR_SCALE}")
    print(f"gan_warmup={GAN_WARMUP_EPOCHS} lambda_adv={LAMBDA_ADV}")
    print(f"use_ref_decode={USE_REF_DECODE}")
    print(f"use_design_pair={USE_DESIGN_PAIR}")
    print(f"use_latent_adv={USE_LATENT_ADV}")
    print(f"select_by_fidelity={SELECT_BY_FIDELITY}")
    print(f"split_by_design={SPLIT_BY_DESIGN}")
    print(f"USE_SAMPLE_MINMAX={USE_SAMPLE_MINMAX} (regen pseudo if changed)")
    print(f"batch_size={BATCH_SIZE}")
    print(f"n_epochs={N_EPOCHS}")
    print(f"learning_rate={LEARNING_RATE}")
    print(f"latent_dim={LATENT_DIM}")
    print(f"input_dim={INPUT_DIM}")
    print(f"early_stopping={EARLY_STOPPING}")
    print(f"patience={PATIENCE}")
    print(f"seed={SEED}")
    print(f"reference_platform={REFERENCE_PLATFORM} (id={REFERENCE_PLATFORM_ID})")
    print("checkpoint: selected by fidelity score -> vae_model.pth")
    print(f"run_gate_after_train={RUN_GATE_AFTER_TRAIN}")
    print("=" * 60)

    run_train(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        latent_dim=LATENT_DIM,
        input_dim=INPUT_DIM,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        seed=SEED,
        use_adversarial=USE_ADVERSARIAL,
        split_by_design=SPLIT_BY_DESIGN,
        use_ref_decode=USE_REF_DECODE,
        use_design_pair=USE_DESIGN_PAIR,
        use_latent_adv=USE_LATENT_ADV,
    )

    if RUN_GATE_AFTER_TRAIN:
        run_gate_evaluation(
            model_path=GATE_MODEL_PATH,
            seed=SEED,
            output_dir=GATE_OUTPUT_DIR,
        )
