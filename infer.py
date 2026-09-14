from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idfb.config import REFERENCE_PLATFORM, REFERENCE_PLATFORM_ID
from idfb.evaluate import run_evaluate
from idfb.integrate import run_integrate


TASK = "MSI"
MODEL_PATH = ROOT / "saved_models" / "vae_model.pth"
USE_SOURCE_PLATFORM = False
REFERENCE_PLATFORM_ID = REFERENCE_PLATFORM_ID
RUN_EVALUATE = True


if __name__ == "__main__":
    print("=" * 60)
    print("IDFB Inference")
    print(f"task={TASK}")
    print(f"model_path={MODEL_PATH}")
    print(f"use_source_platform={USE_SOURCE_PLATFORM}")
    print(f"reference_platform={REFERENCE_PLATFORM} (id={REFERENCE_PLATFORM_ID})")
    print(f"run_evaluate={RUN_EVALUATE}")
    print("=" * 60)

    output_path = run_integrate(
        task=TASK,
        model_path=MODEL_PATH,
        reference_platform_id=REFERENCE_PLATFORM_ID,
        use_source_platform=USE_SOURCE_PLATFORM,
    )
    print(f"Inference output: {output_path}")

    if RUN_EVALUATE:
        run_evaluate(task=TASK)
