import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from idfb.config import (
    GATE_MAX_LATENT_BA,
    GATE_MAX_OUTPUT_BA,
    GATE_MIN_CORR,
    GATE_MIN_CORRECTED_CORR,
    GATE_MIN_CORRECTED_VAR_RATIO,
    GATE_MIN_VAR_RATIO,
    MODEL_DIR,
    N_GPLS,
    REFERENCE_PLATFORM_ID,
    SEED,
)
from idfb.dataset import load_pseudo_arrays
from idfb.models import VAE
from idfb.probe import train_fresh_probe
from idfb.utils import ensure_dir, get_device, set_seed


def pearson_corr_rows(a: np.ndarray, b: np.ndarray) -> float:
    corrs = []
    for i in range(a.shape[0]):
        x = a[i]
        y = b[i]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        corrs.append(np.corrcoef(x, y)[0, 1])
    return float(np.nanmean(corrs)) if corrs else float("nan")


def variance_ratio(x: np.ndarray, y: np.ndarray) -> float:
    vx = float(np.var(x))
    vy = float(np.var(y))
    if vx < 1e-12:
        return float("nan")
    return vy / vx


def load_vae(model_path: Path) -> VAE:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    vae = VAE(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        n_gpls=ckpt["n_gpls"],
    )
    vae.load_state_dict(ckpt["model_state_dict"])
    meta = {
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_gen": ckpt.get("best_val_gen"),
        "best_val_disc": ckpt.get("best_val_disc"),
    }
    vae._gate_meta = meta
    return vae


@torch.no_grad()
def encode_all_with_platform(
    vae: VAE,
    x: np.ndarray,
    p: np.ndarray,
    batch_size: int = 128,
    reference_platform_id: int = REFERENCE_PLATFORM_ID,
    design_ids: Optional[np.ndarray] = None,
    consensus_by_design: bool = False,
) -> Dict[str, np.ndarray]:
    device = get_device()
    vae = vae.to(device)
    vae.eval()

    dataset = TensorDataset(
        torch.tensor(x).float(),
        torch.tensor(p).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    recon_src_list = []
    latent_list = []
    mu_clean_list = []
    p_list = []

    for xb, pb in loader:
        xb = xb.to(device)
        pb = pb.to(device)
        recon_src, mu = vae.reconstruct(xb, pb, deterministic=True, correct=False)
        recon_src_list.append(recon_src.cpu().numpy())
        p_list.append(pb.cpu().numpy())
        if hasattr(vae, "clean_latent"):
            mu_c = vae.clean_latent(mu, pb)
        else:
            mu_c = mu
        mu_clean_list.append(mu_c.cpu().numpy())
        latent_list.append(mu_c.cpu().numpy())

    mu_clean = np.vstack(mu_clean_list)
    if consensus_by_design and design_ids is not None:
        design_ids = np.asarray(design_ids)
        for d in np.unique(design_ids):
            mask = design_ids == d
            if mask.sum() == 0:
                continue
            mu_clean[mask] = mu_clean[mask].mean(axis=0, keepdims=True)

    # Decode corrected from (optionally consensus) cleaned latents.
    recon_ref_list = []
    p_ref_t = torch.full((batch_size,), int(reference_platform_id), device=device)
    for i in range(0, len(mu_clean), batch_size):
        z = torch.tensor(mu_clean[i : i + batch_size]).float().to(device)
        pb = p_ref_t[: z.size(0)]
        recon_ref_list.append(vae.decode(z, pb).cpu().numpy())

    return {
        "recon_source": np.vstack(recon_src_list),
        "recon_corrected": np.vstack(recon_ref_list),
        "latent": np.vstack(latent_list) if not consensus_by_design else mu_clean,
    }


def fidelity_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        "pearson_corr": pearson_corr_rows(x, y),
        "variance_ratio": variance_ratio(x, y),
    }


def design_mean_matrix(x: np.ndarray, design_ids: np.ndarray) -> np.ndarray:
    """Replace each row with the mean of all rows sharing its design_id."""
    out = np.empty_like(x)
    design_ids = np.asarray(design_ids)
    for d in np.unique(design_ids):
        mask = design_ids == d
        out[mask] = x[mask].mean(axis=0, keepdims=True)
    return out


def apply_gate(metrics: Dict) -> Tuple[bool, Dict[str, bool]]:
    # Product is corrected expression (ref-decode ± design consensus).
    corrected_ba = metrics["probe_corrected"]["balanced_accuracy"]
    checks = {
        "corrected_ba_ok": corrected_ba <= GATE_MAX_OUTPUT_BA,
        "latent_ba_ok": metrics["probe_latent"]["balanced_accuracy"]
        <= GATE_MAX_LATENT_BA,
        "corr_ok": metrics["fidelity_recon_source"]["pearson_corr"] >= GATE_MIN_CORR,
        "var_ok": metrics["fidelity_recon_source"]["variance_ratio"]
        >= GATE_MIN_VAR_RATIO,
        "corrected_corr_ok": metrics["fidelity_recon_corrected"]["pearson_corr"]
        >= GATE_MIN_CORRECTED_CORR,
        "corrected_var_ok": metrics["fidelity_recon_corrected"]["variance_ratio"]
        >= GATE_MIN_CORRECTED_VAR_RATIO,
    }
    return all(checks.values()), checks


def run_gate_evaluation(
    model_path: Optional[Path] = None,
    seed: int = SEED,
    batch_size: int = 128,
    output_dir: Optional[Path] = None,
    consensus_by_design: bool = True,
    max_samples: int = 8000,
) -> Dict:
    set_seed(seed)
    if model_path is None:
        model_path = MODEL_DIR / "vae_model.pth"
    if output_dir is None:
        output_dir = MODEL_DIR / "gate"
    ensure_dir(output_dir)

    x, p, design_ids = load_pseudo_arrays()
    if max_samples and len(x) > max_samples:
        rng = np.random.default_rng(seed)
        # keep whole designs so consensus_by_design still has partners
        uniq = np.unique(design_ids)
        rng.shuffle(uniq)
        chosen = []
        n = 0
        for d in uniq:
            m = int((design_ids == d).sum())
            if chosen and n + m > max_samples:
                break
            chosen.append(d)
            n += m
        mask = np.isin(design_ids, np.asarray(chosen))
        x, p, design_ids = x[mask], p[mask], design_ids[mask]
        print(f"Gate subsample: n={len(x)} designs={len(chosen)}", flush=True)

    vae = load_vae(Path(model_path))
    reps = encode_all_with_platform(
        vae,
        x,
        p,
        batch_size=batch_size,
        design_ids=design_ids,
        consensus_by_design=consensus_by_design,
    )

    probe_input = train_fresh_probe(x, p, seed=seed, use_pca=True)
    probe_output = train_fresh_probe(
        reps["recon_source"], p, seed=seed + 1, use_pca=True
    )
    probe_corrected = train_fresh_probe(
        reps["recon_corrected"], p, seed=seed + 2, use_pca=True
    )
    probe_latent = train_fresh_probe(
        reps["latent"], p, seed=seed + 3, use_pca=False
    )

    fid_src = fidelity_metrics(x, reps["recon_source"])
    # With design consensus, corrected should match cross-platform design mean biology.
    if consensus_by_design:
        x_des = design_mean_matrix(x, design_ids)
        fid_corr = fidelity_metrics(x_des, reps["recon_corrected"])
    else:
        fid_corr = fidelity_metrics(x, reps["recon_corrected"])

    metrics = {
        "model_path": str(model_path),
        "n_samples": int(x.shape[0]),
        "n_platforms": int(N_GPLS),
        "chance_level": 1.0 / N_GPLS,
        "consensus_by_design": consensus_by_design,
        "checkpoint_meta": getattr(vae, "_gate_meta", {}),
        "thresholds": {
            "GATE_MAX_OUTPUT_BA": GATE_MAX_OUTPUT_BA,
            "GATE_MAX_LATENT_BA": GATE_MAX_LATENT_BA,
            "GATE_MIN_CORR": GATE_MIN_CORR,
            "GATE_MIN_VAR_RATIO": GATE_MIN_VAR_RATIO,
            "GATE_MIN_CORRECTED_CORR": GATE_MIN_CORRECTED_CORR,
            "GATE_MIN_CORRECTED_VAR_RATIO": GATE_MIN_CORRECTED_VAR_RATIO,
        },
        "probe_input": probe_input,
        "probe_output": probe_output,
        "probe_corrected": probe_corrected,
        "probe_latent": probe_latent,
        "fidelity_recon_source": fid_src,
        "fidelity_recon_corrected": fid_corr,
    }
    passed, checks = apply_gate(metrics)
    metrics["gate_checks"] = checks
    metrics["gate_passed"] = passed

    out_json = output_dir / "gate_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        {
            "name": "input",
            "balanced_accuracy": probe_input["balanced_accuracy"],
            "macro_f1": probe_input["macro_f1"],
            "mean_entropy": probe_input["mean_entropy"],
            "pearson_corr": 1.0,
            "variance_ratio": 1.0,
        },
        {
            "name": "recon_source",
            "balanced_accuracy": probe_output["balanced_accuracy"],
            "macro_f1": probe_output["macro_f1"],
            "mean_entropy": probe_output["mean_entropy"],
            "pearson_corr": fid_src["pearson_corr"],
            "variance_ratio": fid_src["variance_ratio"],
        },
        {
            "name": "recon_corrected",
            "balanced_accuracy": probe_corrected["balanced_accuracy"],
            "macro_f1": probe_corrected["macro_f1"],
            "mean_entropy": probe_corrected["mean_entropy"],
            "pearson_corr": fid_corr["pearson_corr"],
            "variance_ratio": fid_corr["variance_ratio"],
        },
        {
            "name": "latent",
            "balanced_accuracy": probe_latent["balanced_accuracy"],
            "macro_f1": probe_latent["macro_f1"],
            "mean_entropy": probe_latent["mean_entropy"],
            "pearson_corr": float("nan"),
            "variance_ratio": float("nan"),
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "gate_summary.csv", index=False)

    print("=" * 60)
    print("IDFB post-training diagnostic with fresh probes")
    print(f"model: {model_path}")
    print(f"consensus_by_design={consensus_by_design}")
    print(f"chance_level: {1.0 / N_GPLS:.4f}")
    print(summary.to_string(index=False))
    print("-" * 60)
    for k, v in checks.items():
        print(f"{k}: {v}")
    print(f"HARD_GATE_PASSED: {passed}")
    print(f"saved: {out_json}")
    print("=" * 60)
    return metrics
