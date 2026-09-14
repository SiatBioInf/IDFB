from idfb.config import PSEUDO_NUM_SAMPLES
from idfb.pseudo import run_generate_pseudo
from idfb.tasks.cancertype import save_cancertype_processed
from idfb.tasks.lung_cancer import save_lung_cancer_processed
from idfb.tasks.msi import save_msi_processed
from idfb.tasks.survival import save_survival_processed


TASK_HANDLERS = {
    "MSI": save_msi_processed,
    "Cancertype": save_cancertype_processed,
    "Lung_cancer_subtybes": save_lung_cancer_processed,
    "Survival_analysis": save_survival_processed,
}


def run_preprocess(
    tasks=None,
    with_pseudo=True,
    num_samples=PSEUDO_NUM_SAMPLES,
    seed=42,
):
    if tasks is None:
        tasks = list(TASK_HANDLERS.keys())

    results = {}
    if with_pseudo:
        print("\n########## Generate pseudo data ##########")
        run_generate_pseudo(num_samples=num_samples, seed=seed)
        results["pseudo"] = True

    for task in tasks:
        if task not in TASK_HANDLERS:
            raise ValueError(f"Unknown task: {task}")
        print(f"\n########## Prepare task: {task} ##########")
        out = TASK_HANDLERS[task]()
        results[task] = str(out)

    print("\n########## Preprocess finished ##########")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results
