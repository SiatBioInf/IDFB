import argparse
from pathlib import Path

import torch  # noqa: F401  Windows: import torch before numpy
import numpy as np
import pandas as pd

from idfb.baselines import combat, sva_correct
from idfb.config import (
    DATASET_DIR,
    INTEGRATE_ALIGN_PLATFORM_MEANS,
    INTEGRATE_BIAS_SCALE,
    INTEGRATE_CLASS_SCALE,
    INTEGRATE_DELTA_CORRECT,
    INTEGRATE_PC_DROP,
    INTEGRATE_PC_MIX,
    INTEGRATE_PC_N,
    INTEGRATE_PLATFORM_SV_DROP,
    INTEGRATE_PLATFORM_SV_K,
    INTEGRATE_PLATFORM_SV_MIX,
    INTEGRATE_PROTECT_LABELS,
    INTEGRATE_REMOVE_PLATFORM_PCS,
    INTEGRATE_RESIDUAL_COMBAT,
    INTEGRATE_RESIDUAL_SVA,
    INTEGRATE_TASK_OVERRIDES,
    INTEGRATE_SVA_BLEND,
    INTEGRATE_SVA_EXTRA,
    INTEGRATE_SVA_SV,
    INTEGRATE_WITHIN_CLASS_ALIGN,
    INTEGRATE_WITHIN_CLASS_PC_MIX,
    MODEL_DIR,
    REFERENCE_PLATFORM,
    REFERENCE_PLATFORM_ID,
)
from idfb.dataset import load_task_tensors
from idfb.models import VAE
from idfb.utils import ensure_dir, get_device


def load_vae_model(model_path: Path) -> VAE:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run training first."
        )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    vae = VAE(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
        n_gpls=checkpoint["n_gpls"],
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    return vae


def align_group_means(
    arr: np.ndarray,
    groups: np.ndarray,
    reference_id: int,
) -> np.ndarray:
    """Shift each group so its mean matches the reference group mean."""
    out = np.asarray(arr, dtype=np.float64).copy()
    g = np.asarray(groups).astype(np.int64)
    ref_mask = g == int(reference_id)
    if int(ref_mask.sum()) >= 1:
        ref_mean = out[ref_mask].mean(axis=0)
    else:
        ref_mean = out.mean(axis=0)
    for gid in np.unique(g):
        mask = g == gid
        if int(mask.sum()) == 0:
            continue
        out[mask] = out[mask] - out[mask].mean(axis=0) + ref_mean
    return out.astype(np.float32)


def remove_platform_components(
    arr: np.ndarray,
    platforms: np.ndarray,
    n_components: int = 8,
) -> np.ndarray:
    """Remove the PCs most associated with platform (ANOVA F), then reconstruct."""
    from scipy.stats import f_oneway
    from sklearn.decomposition import PCA

    x = np.asarray(arr, dtype=np.float64)
    p = np.asarray(platforms).astype(np.int64)
    n_plat = len(np.unique(p))
    n_components = int(min(n_components, x.shape[0] - 1, x.shape[1]))
    if n_components < 1 or n_plat < 2:
        return x.astype(np.float32)

    x_c = x - x.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(x_c)

    f_stats = []
    for i in range(n_components):
        groups = [scores[p == plat, i] for plat in np.unique(p) if (p == plat).sum() > 1]
        if len(groups) < 2:
            f_stats.append(0.0)
            continue
        try:
            f_stats.append(float(f_oneway(*groups).statistic))
        except Exception:
            f_stats.append(0.0)

    # Drop the strongest platform-linked PCs (at least half of them).
    order = np.argsort(f_stats)[::-1]
    n_drop = max(1, n_components // 2)
    drop = order[:n_drop].tolist()
    drop_scores = scores[:, drop]
    drop_loadings = pca.components_[drop]
    residual = x_c - drop_scores @ drop_loadings
    return (residual + x.mean(axis=0, keepdims=True)).astype(np.float32)


def match_group_moments(
    arr: np.ndarray,
    groups: np.ndarray,
    reference_id: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Match per-group mean and std to the reference group (gene-wise)."""
    out = np.asarray(arr, dtype=np.float64).copy()
    g = np.asarray(groups).astype(np.int64)
    ref_mask = g == int(reference_id)
    if int(ref_mask.sum()) < 2:
        return align_group_means(out, g, reference_id)
    ref_mean = out[ref_mask].mean(axis=0)
    ref_std = out[ref_mask].std(axis=0) + eps
    for gid in np.unique(g):
        mask = g == gid
        if int(mask.sum()) < 2:
            continue
        mu = out[mask].mean(axis=0)
        sd = out[mask].std(axis=0) + eps
        out[mask] = (out[mask] - mu) / sd * ref_std + ref_mean
    return out.astype(np.float32)


def align_means_within_class(
    arr: np.ndarray,
    platforms: np.ndarray,
    labels: np.ndarray,
    reference_id: int,
) -> np.ndarray:
    """Shift platform means inside each biological class (no std matching)."""
    out = np.asarray(arr, dtype=np.float64).copy()
    p = np.asarray(platforms)
    y = np.asarray(labels)
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if len(idx) < 4:
            continue
        out[idx] = align_group_means(out[idx], p[idx], reference_id)
    return out.astype(np.float32)


def _platform_pc_fstats(scores: np.ndarray, platforms: np.ndarray) -> np.ndarray:
    from scipy.stats import f_oneway

    p = np.asarray(platforms)
    plats = np.unique(p)
    f_stats = np.zeros(scores.shape[1], dtype=np.float64)
    for i in range(scores.shape[1]):
        groups = [scores[p == plat, i] for plat in plats if int((p == plat).sum()) > 1]
        if len(groups) < 2:
            continue
        try:
            f_stats[i] = float(f_oneway(*groups).statistic)
        except Exception:
            f_stats[i] = 0.0
    return f_stats


def remove_platform_pcs_within_class(
    arr: np.ndarray,
    platforms: np.ndarray,
    labels: np.ndarray,
    n_components: int = 12,
    n_drop: int = 6,
    mix: float = 0.55,
) -> np.ndarray:
    """Drop platform-linked PCs inside each class; leave between-class biology."""
    from sklearn.decomposition import PCA

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-8 or int(n_drop) < 1:
        return np.asarray(arr, dtype=np.float32)
    out = np.asarray(arr, dtype=np.float64).copy()
    p = np.asarray(platforms)
    y = np.asarray(labels)
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if len(idx) < 12:
            continue
        p_sub = p[idx]
        if len(np.unique(p_sub)) < 2:
            continue
        sub = out[idx]
        n_comp = int(min(n_components, len(idx) - 2, sub.shape[1], 20))
        n_rm = int(min(max(n_drop, 1), n_comp))
        if n_comp < 2:
            continue
        mu = sub.mean(axis=0, keepdims=True)
        xc = sub - mu
        pca = PCA(n_components=n_comp, random_state=0)
        scores = pca.fit_transform(xc)
        order = np.argsort(_platform_pc_fstats(scores, p_sub))[::-1]
        drop = order[:n_rm]
        removed = scores[:, drop] @ pca.components_[drop]
        out[idx] = sub - mix * removed
    return out.astype(np.float32)


def remove_platform_linked_svs(
    arr: np.ndarray,
    platforms: np.ndarray,
    labels: np.ndarray,
    n_sv: int = 12,
    n_drop: int = 6,
    mix: float = 0.45,
) -> np.ndarray:
    """Remove residual SVs most associated with platform, labels protected."""
    from sklearn.decomposition import TruncatedSVD

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-8 or int(n_drop) < 1:
        return np.asarray(arr, dtype=np.float32)
    x = np.asarray(arr, dtype=np.float64)
    y = np.asarray(labels)
    p = np.asarray(platforms)
    dummies = pd.get_dummies(pd.Series(y), drop_first=True)
    prot = (
        dummies.to_numpy(dtype=np.float64)
        if dummies.shape[1] > 0
        else np.zeros((x.shape[0], 0), dtype=np.float64)
    )
    c = np.hstack([np.ones((x.shape[0], 1)), prot])
    coef, *_ = np.linalg.lstsq(c, x, rcond=None)
    resid = x - c @ coef
    k = int(min(max(n_sv, 1), x.shape[0] - 2, x.shape[1] - 1, 20))
    n_rm = int(min(max(n_drop, 1), k))
    sv = TruncatedSVD(n_components=k, random_state=0).fit_transform(resid)
    order = np.argsort(_platform_pc_fstats(sv, p))[::-1]
    drop = order[:n_rm]
    sv_d = sv[:, drop]
    c2 = np.hstack([c, sv_d])
    coef2, *_ = np.linalg.lstsq(c2, x, rcond=None)
    delta = sv_d @ coef2[c.shape[1] :]
    return (x - mix * delta).astype(np.float32)


def expand_class_means(arr: np.ndarray, labels: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Move class centroids apart after mixing, without changing within-class geometry."""
    scale = float(scale)
    out = np.asarray(arr, dtype=np.float64).copy()
    if abs(scale - 1.0) < 1e-8:
        return out.astype(np.float32)
    y = np.asarray(labels)
    g = out.mean(axis=0)
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if len(idx) < 2:
            continue
        mu = out[idx].mean(axis=0)
        out[idx] = out[idx] + (scale - 1.0) * (mu - g)
    return out.astype(np.float32)


def generate_corrected(
    vae: VAE,
    x: torch.Tensor,
    p_source: torch.Tensor,
    reference_platform_id: int,
    y_labels=None,
    align_means: bool = True,
    bias_scale: float = 1.0,
    remove_pcs: int = 0,
    protect_labels: bool = True,
    residual_combat: bool = False,
    residual_sva: bool = False,
    n_sv: int = 8,
    within_class: bool = True,
    delta_correct: bool = True,
    within_class_pc_mix: bool = False,
    pc_n: int = 12,
    pc_drop: int = 6,
    pc_mix: float = 0.0,
    platform_sv_mix: float = 0.0,
    platform_sv_k: int = 12,
    platform_sv_drop: int = 6,
    sva_blend: float = 0.0,
    sva_extra: float = 0.0,
    class_scale: float = 1.0,
):
    """Counterfactual platform shift as a residual on x, then class-protected mixing."""
    device = get_device()
    vae.eval()
    vae = vae.to(device)
    with torch.no_grad():
        x0 = x.to(device)
        scale = x0.abs().amax(dim=0).clamp_min(1e-8)
        xb = x0 / scale
        pb = p_source.to(device).clamp(min=0, max=vae.n_gpls - 1)
        loc = vae.encode_mu(xb)
        z = loc - float(bias_scale) * vae.platform_bias(pb)
        p_ref = torch.full_like(pb, int(reference_platform_id))
        recon_ref = vae.decode(z, p_ref) * scale
        if delta_correct:
            recon_src = vae.decode(loc, pb) * scale
            recon = x0 + (recon_ref - recon_src)
        else:
            recon = recon_ref
    arr = recon.cpu().numpy().astype(np.float64)
    p_np = pb.cpu().numpy()
    y_np = None if y_labels is None else np.asarray(y_labels).reshape(-1)
    protect = y_np if (protect_labels and y_np is not None) else None
    if residual_combat:
        arr = combat(arr, p_np, protect=protect)
    if residual_sva and protect is not None:
        arr = sva_correct(arr, protect=protect, n_sv=int(n_sv))
    if within_class and y_np is not None:
        arr = align_means_within_class(
            arr, p_np, y_np, int(reference_platform_id)
        )
    elif align_means:
        arr = align_group_means(arr, p_np, int(reference_platform_id))
    if within_class_pc_mix and y_np is not None:
        arr = remove_platform_pcs_within_class(
            arr,
            p_np,
            y_np,
            n_components=int(pc_n),
            n_drop=int(pc_drop),
            mix=float(pc_mix),
        )
    if float(platform_sv_mix) > 0.0 and y_np is not None:
        arr = remove_platform_linked_svs(
            arr,
            p_np,
            y_np,
            n_sv=int(platform_sv_k),
            n_drop=int(platform_sv_drop),
            mix=float(platform_sv_mix),
        )
    if float(sva_blend) > 0.0 and protect is not None:
        xs = sva_correct(arr, protect=protect, n_sv=int(n_sv))
        a = float(np.clip(sva_blend, 0.0, 1.0))
        arr = (1.0 - a) * np.asarray(arr, dtype=np.float64) + a * xs
    if float(sva_extra) > 0.0 and protect is not None:
        xs = sva_correct(arr, protect=protect, n_sv=int(n_sv))
        a = float(np.clip(sva_extra, 0.0, 1.0))
        arr = (1.0 - a) * np.asarray(arr, dtype=np.float64) + a * xs
    if y_np is not None and abs(float(class_scale) - 1.0) > 1e-8:
        arr = expand_class_means(arr, y_np, scale=float(class_scale))
    if int(remove_pcs) > 0:
        arr = remove_platform_components(arr, p_np, n_components=int(remove_pcs))
    return arr.astype(np.float32)


def generate_recon(vae: VAE, x: torch.Tensor, p: torch.Tensor):
    device = get_device()
    vae.eval()
    vae = vae.to(device)
    with torch.no_grad():
        xb = x.to(device)
        scale = xb.abs().amax(dim=0).clamp_min(1e-8)
        pb = p.to(device).clamp(min=0, max=vae.n_gpls - 1)
        recon, _, _ = vae(xb / scale, pb)
        recon = recon * scale
    return recon.cpu().numpy()


_INTEGRATE_OVERRIDE_KEYS = frozenset(
    {
        "residual_combat",
        "residual_sva",
        "platform_sv_mix",
        "platform_sv_k",
        "platform_sv_drop",
        "sva_blend",
        "sva_extra",
        "within_class_pc_mix",
        "pc_n",
        "pc_drop",
        "pc_mix",
        "n_sv",
        "class_scale",
    }
)


def _resolve_integrate_params(task: str) -> dict:
    overrides = dict(INTEGRATE_TASK_OVERRIDES.get(task, {}))
    unknown = set(overrides) - _INTEGRATE_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"Unknown INTEGRATE_TASK_OVERRIDES keys for {task}: {sorted(unknown)}"
        )

    def pick(key, default):
        return overrides.pop(key, default)

    return {
        "residual_combat": bool(pick("residual_combat", INTEGRATE_RESIDUAL_COMBAT)),
        "residual_sva": bool(pick("residual_sva", INTEGRATE_RESIDUAL_SVA)),
        "platform_sv_mix": float(pick("platform_sv_mix", INTEGRATE_PLATFORM_SV_MIX)),
        "platform_sv_k": int(pick("platform_sv_k", INTEGRATE_PLATFORM_SV_K)),
        "platform_sv_drop": int(
            pick("platform_sv_drop", INTEGRATE_PLATFORM_SV_DROP)
        ),
        "sva_blend": float(pick("sva_blend", INTEGRATE_SVA_BLEND)),
        "sva_extra": float(pick("sva_extra", INTEGRATE_SVA_EXTRA)),
        "within_class_pc_mix": bool(
            pick("within_class_pc_mix", INTEGRATE_WITHIN_CLASS_PC_MIX)
        ),
        "pc_n": int(pick("pc_n", INTEGRATE_PC_N)),
        "pc_drop": int(pick("pc_drop", INTEGRATE_PC_DROP)),
        "pc_mix": float(pick("pc_mix", INTEGRATE_PC_MIX)),
        "n_sv": int(pick("n_sv", INTEGRATE_SVA_SV)),
        "class_scale": float(pick("class_scale", INTEGRATE_CLASS_SCALE)),
    }


def run_integrate(
    task: str,
    model_path: Path = None,
    reference_platform_id: int = None,
    use_source_platform: bool = False,
    align_platform_means_flag: bool = None,
    bias_scale: float = None,
):
    if model_path is None:
        model_path = MODEL_DIR / "vae_model.pth"
    if reference_platform_id is None:
        reference_platform_id = REFERENCE_PLATFORM_ID
    if align_platform_means_flag is None:
        align_platform_means_flag = INTEGRATE_ALIGN_PLATFORM_MEANS
    if bias_scale is None:
        bias_scale = float(INTEGRATE_BIAS_SCALE)
    remove_pcs = int(INTEGRATE_REMOVE_PLATFORM_PCS)

    data_path = DATASET_DIR / task / "processed_data.csv"
    output_path = DATASET_DIR / task / "generated_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Prepare task data first."
        )

    ensure_dir(output_path.parent)
    vae = load_vae_model(Path(model_path))
    x, p_source, y_lab = load_task_tensors(data_path)

    if use_source_platform:
        p_decode = p_source.clamp(min=0, max=vae.decode.n_gpls - 1)
        print("Decode platform: source platform labels")
        recon = generate_recon(vae, x, p_decode)
    else:
        params = _resolve_integrate_params(task)
        print(
            f"Decode platform: reference={REFERENCE_PLATFORM} "
            f"(id={reference_platform_id}) via z_clean, "
            f"delta_correct={INTEGRATE_DELTA_CORRECT}, "
            f"bias_scale={bias_scale}, within_class={INTEGRATE_WITHIN_CLASS_ALIGN}, "
            f"residual_combat={params['residual_combat']}, "
            f"residual_sva={params['residual_sva']}, "
            f"platform_sv_mix={params['platform_sv_mix']}, "
            f"pc_mix={params['pc_mix'] if params['within_class_pc_mix'] else 0}, "
            f"sva_blend={params['sva_blend']}, sva_extra={params['sva_extra']}, "
            f"class_scale={params['class_scale']}"
        )
        recon = generate_corrected(
            vae,
            x,
            p_source,
            reference_platform_id,
            y_labels=y_lab.numpy(),
            align_means=align_platform_means_flag,
            bias_scale=bias_scale,
            remove_pcs=remove_pcs,
            protect_labels=INTEGRATE_PROTECT_LABELS,
            within_class=INTEGRATE_WITHIN_CLASS_ALIGN,
            delta_correct=INTEGRATE_DELTA_CORRECT,
            **params,
        )

    pd.DataFrame(recon).to_csv(output_path, index=False)
    print(f"Generated shape: {recon.shape}")
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Integrate task data with IDFB")
    parser.add_argument("--task", type=str, default="MSI")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--use-source-platform", action="store_true")
    parser.add_argument(
        "--no-align-means",
        action="store_true",
        help="Disable post-correction platform moment matching",
    )
    parser.add_argument(
        "--bias-scale",
        type=float,
        default=None,
        help="Scale for subtracting platform_bias at inference",
    )
    args = parser.parse_args()
    model_path = Path(args.model) if args.model else None
    run_integrate(
        args.task,
        model_path,
        use_source_platform=args.use_source_platform,
        align_platform_means_flag=(False if args.no_align_means else None),
        bias_scale=args.bias_scale,
    )


if __name__ == "__main__":
    main()
