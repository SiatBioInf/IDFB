from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"
MODEL_DIR = ROOT / "saved_models"
HEADER_PATH = DATASET_DIR / "header.txt"

PSEUDO_DIR = DATASET_DIR / "Pseudo data"
PSEUDO_SOURCE_DIR = DATASET_DIR / "Pseudo_data_生成伪数据集的细胞系数据"

PSEUDO_PLATFORM_SOURCES = {
    "GPL570": PSEUDO_SOURCE_DIR / "Affymetrix" / "GPL570",
    "GPL13497": PSEUDO_SOURCE_DIR / "Agilent" / "GPL13497",
    "GPL20301": PSEUDO_SOURCE_DIR / "Illumina" / "GPL20301",
    "GPL24676": PSEUDO_SOURCE_DIR / "Illumina" / "GPL24676",
}

TASK_SOURCES = {
    "MSI": DATASET_DIR / "MSI_结直肠癌MSI亚型原发灶数据",
    "Cancertype": DATASET_DIR / "Cancertype_肿瘤原发灶数据",
    "Lung_cancer_subtybes": DATASET_DIR / "Lung_cancer_subtybes_肺癌亚型原发灶数据",
    "Survival_analysis": DATASET_DIR / "Survival_analysis_生存分析数据",
}

TASK_OUTPUTS = {
    "MSI": DATASET_DIR / "MSI",
    "Cancertype": DATASET_DIR / "Cancertype",
    "Lung_cancer_subtybes": DATASET_DIR / "Lung_cancer_subtybes",
    "Survival_analysis": DATASET_DIR / "Survival_analysis",
}

PLATFORMS = ["GPL570", "GPL13497", "GPL20301", "GPL24676"]
REFERENCE_PLATFORM = "GPL570"
PLATFORM_ENCODER = {gpl: idx for idx, gpl in enumerate(PLATFORMS)}
REFERENCE_PLATFORM_ID = PLATFORM_ENCODER[REFERENCE_PLATFORM]
N_GPLS = len(PLATFORMS)

CELL_TYPES = ["细胞系", "巨噬细胞", "B细胞", "T细胞"]

INPUT_DIM = 19871
LATENT_DIM = 512
BATCH_SIZE = 64
N_EPOCHS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 512
TRAIN_TEST_RATIO = 0.2
# Paper-scale shared designs: 45000 × 4 platforms ≈ 180000 pseudo-bulk samples.
PSEUDO_NUM_SAMPLES = 45000
SEED = 42

# The expression discriminator is activated after a warm-up period and updated
# less frequently than the VAE. Unstable adversarial steps are not retained.
TRAIN_USE_ADVERSARIAL = True
# Per-sample MinMax removes amplitude; default off. Regenerate pseudo after changing.
USE_SAMPLE_MINMAX = False
# Keep the same design_id on one side of the split (all platforms of a design).
SPLIT_BY_DESIGN = True

# Reconstruction: Huber + gene down-weighting (heteroscedastic bulk / high-expr domination).
RECON_LOSS = "huber"  # "mse" | "huber"
HUBER_DELTA = 0.5
LAMBDA_RECON_CORR = 0.20
GENE_WEIGHT_POWER = 0.5  # weight ∝ 1 / (mean_abs^power + eps)
KL_BETA_MAX = 0.02
KL_WARMUP_EPOCHS = 20
KL_MAX = 5.0

# Reference-platform decode + design pairing (biology-preserving correction).
USE_REF_DECODE_LOSS = True
USE_DESIGN_PAIR_LOSS = True
# Avoid hard MSE(ref, raw x): that writes platform fingerprints into corrected output.
LAMBDA_REF_MSE = 0.0
LAMBDA_REF_SRC = 0.25
LAMBDA_REF_VAR = 0.25
LAMBDA_REF_CORR = 0.0
# Soft hinge: keep pearson(corrected, x) above target (biology proxy).
LAMBDA_FID_HINGE = 2.0
FID_HINGE_TARGET = 0.70
LAMBDA_DESIGN_Z = 0.50
LAMBDA_DESIGN_REF = 0.40
LAMBDA_DESIGN_CENTER = 0.25
# Collapse cleaned latents within each design (platform-invariant z).
LAMBDA_Z_WITHIN = 0.50
# Pull per-platform means of z_clean together (helps unpaired mixing / SAS).
LAMBDA_PLAT_MEAN = 0.20
LAMBDA_MMD = 0.40
USE_LATENT_ADV = True
LAMBDA_LATENT_ADV = 0.03
LATENT_ADV_WARMUP_EPOCHS = 30
GRL_LAMBDA = 0.3
ADV_ON_CORRECTED = True
LAMBDA_ADV = 0.02
GAN_WARMUP_EPOCHS = 40
DISC_LR_SCALE = 0.25
DISC_UPDATE_EVERY = 3
GRAD_CLIP_NORM = 1.0
MAX_GEN_LOSS = 2.5
SELECT_BY_FIDELITY = True
SELECT_LEAK_PENALTY = 0.0
SELECT_MIN_CORR = 0.55
SELECT_MAX_VAL_RATIO = 1.50
INTEGRATE_ALIGN_PLATFORM_MEANS = True
INTEGRATE_BIAS_SCALE = 1.0
INTEGRATE_REMOVE_PLATFORM_PCS = 0
# Apply platform shift as residual: x + (decode_ref - decode_source).
# Replacing x with decode_ref destroys kNN biology (low NC).
INTEGRATE_DELTA_CORRECT = True
# Label-protected mean shift only (no std matching, no residual ComBat/SVA).
INTEGRATE_PROTECT_LABELS = True
INTEGRATE_RESIDUAL_COMBAT = False
INTEGRATE_RESIDUAL_SVA = False
# Per-task overrides for integration.
# platform_sv_mix: label-protected removal of platform-linked surrogate variables
# (SVA-like, partial strength; 0 = off). Not full SVA blend.
INTEGRATE_TASK_OVERRIDES = {
    "MSI": {
        "residual_combat": True,
        "platform_sv_mix": 0.45,
        "platform_sv_k": 12,
        "platform_sv_drop": 5,
    },
    "Lung_cancer_subtybes": {
        "platform_sv_mix": 0.30,
        "platform_sv_k": 12,
        "platform_sv_drop": 5,
    },
}
INTEGRATE_WITHIN_CLASS_ALIGN = True
# Optional within-class platform-PC removal (disabled by default).
INTEGRATE_WITHIN_CLASS_PC_MIX = False
INTEGRATE_PC_N = 12
INTEGRATE_PC_DROP = 8
INTEGRATE_PC_MIX = 0.0
# Optional biology-protected, platform-linked SV residual (blend; 0 = off).
INTEGRATE_PLATFORM_SV_MIX = 0.0
INTEGRATE_PLATFORM_SV_K = 12
INTEGRATE_PLATFORM_SV_DROP = 6
# Do not blend SVA into IDFB output.
INTEGRATE_SVA_BLEND = 0.0
INTEGRATE_SVA_SV = 8
INTEGRATE_SVA_EXTRA = 0.0
# Do not expand class centroids after correction.
INTEGRATE_CLASS_SCALE = 1.0

# Hard success thresholds (4 platforms, chance ≈ 0.25). All must pass.
GATE_MAX_OUTPUT_BA = 0.35
GATE_MAX_LATENT_BA = 0.35
GATE_MIN_CORR = 0.70
GATE_MIN_VAR_RATIO = 0.50
GATE_MIN_CORRECTED_CORR = 0.70
GATE_MIN_CORRECTED_VAR_RATIO = 0.50
GATE_PROBE_PCA_DIM = 64
GATE_PROBE_TEST_SIZE = 0.3

TASKS = {
    "MSI": "MSI",
    "Cancertype": "Cancertype",
    "Lung_cancer_subtybes": "Lung_cancer_subtybes",
    "Survival_analysis": "Survival_analysis",
    "Demo": "Demo",
}

CANCER_TYPES = ["liver", "lung", "colorectal", "pancreatic"]
LUNG_SUBTYPES = ["非小细胞肺癌", "小细胞肺癌", "肺腺癌细胞系"]
