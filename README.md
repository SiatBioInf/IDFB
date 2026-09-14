# IDFB

IDFB integrates bulk transcriptomic profiles measured on different GPL platforms. The current implementation trains a conditional VAE with adversarial platform constraints on paired pseudo-bulk mixtures. At inference, it applies paired-delta correction and decodes toward the GPL570 reference condition.

## Repository contents

- `idfb/`: model, training, integration, evaluation and data-preparation modules.
- `train.py`: the supplied configuration for model training and post-training diagnostics.
- `infer.py`: example integration entry point.
- `gate_eval.py`: post-training platform-leakage and expression-fidelity diagnostics.
- `scripts/run_benchmark.py`: nine-method task comparison and DEG overlap analysis.
- `scripts/run_paper_eval.py`: six-method comparison used for the manuscript metrics.
- `Dataset/header.txt`: ordered list of the 19,871 model input genes.

Plotting and presentation scripts are not included in this release. Large expression matrices, trained weights and generated results are distributed separately.

## Installation

Python 3.8 or later is required. Install PyTorch for the CUDA version available on the target machine, then install IDFB in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

For a CPU-only or different CUDA installation, follow the corresponding command at the PyTorch website. The remaining pinned dependencies are listed in `requirements.txt`.

## Input layout

The repository expects data beneath `Dataset/`. Expression matrices must use the gene order in `Dataset/header.txt`.

```text
Dataset/
├── header.txt
├── Pseudo_data_生成伪数据集的细胞系数据/
│   ├── Affymetrix/GPL570/
│   ├── Agilent/GPL13497/
│   ├── Illumina/GPL20301/
│   └── Illumina/GPL24676/
├── Pseudo data/
├── MSI/
├── Cancertype/
├── Lung_cancer_subtybes/
└── Survival_analysis/
```

Each downstream `processed_data.csv` contains the 19,871 expression columns followed by a GPL label column and a task-label column. The preprocessing modules read the study-specific source folders defined in `idfb/config.py`; those source matrices are not included in this repository.

## Pseudo-bulk generation and training

Pseudo-bulk samples use shared mixture designs across GPL platforms. Generate them and train the model with:

```bash
idfb generate-pseudo --num-samples 45000 --seed 42
python train.py
```

The main settings are defined in `idfb/config.py`. The paper configuration uses a 512-dimensional latent representation, Huber reconstruction loss, KL warm-up, delayed discriminator updates and GPL570 as the reference platform. Checkpoints and diagnostic outputs are written below `saved_models/`.

The diagnostic script can also be run separately:

```bash
python gate_eval.py
```

Its fresh classifiers measure residual GPL predictability in the saved representation and corrected expression. These diagnostics do not constitute an independent external validation set.

## Integration and evaluation

Prepare a downstream task, generate corrected expression and calculate the integration metrics:

```bash
idfb prepare --task MSI
idfb integrate --task MSI --model saved_models/vae_model.pth
idfb evaluate --task MSI
```

Valid task names are `MSI`, `Cancertype`, `Lung_cancer_subtybes` and `Survival_analysis`.

To reproduce the supplied method-comparison tables after all task matrices and IDFB outputs have been prepared:

```bash
python scripts/run_benchmark.py
python scripts/run_paper_eval.py
```

Some comparison methods are implemented as bulk-data or PCA-space approximations because their published software targets single-cell objects. See `idfb/baselines.py` for the implementation used for each method.

## Reproducibility notes

The public code begins with platform-preprocessed expression matrices; it does not perform CEL or FASTQ processing. The full pseudo-bulk and patient matrices are required to reproduce the reported results. Because the trained checkpoint is not stored in the Git repository, place the supplied `vae_model.pth` in `saved_models/` before inference.
