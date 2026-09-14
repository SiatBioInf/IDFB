"""Bulk / cross-platform correction baselines (gene-level matrices)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD


def _one_hot(labels, drop_first: bool = True) -> np.ndarray:
    dummies = pd.get_dummies(pd.Series(labels), drop_first=drop_first)
    if dummies.shape[1] == 0:
        return np.zeros((len(labels), 0), dtype=np.float64)
    return dummies.to_numpy(dtype=np.float64)


def _lstsq_coef(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(C, X, rcond=None)
    return coef


def uncorrected(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).copy()


def limma_remove_batch(X: np.ndarray, batch, protect=None) -> np.ndarray:
    """OLS analogue of limma::removeBatchEffect (keep biology as covariate)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    prot = _one_hot(protect, drop_first=True) if protect is not None else np.zeros((n, 0))
    bat = _one_hot(batch, drop_first=True)
    C = np.hstack([np.ones((n, 1)), prot, bat])
    n_keep = 1 + prot.shape[1]
    coef = _lstsq_coef(C, X)
    return X - C[:, n_keep:] @ coef[n_keep:]


def combat(X: np.ndarray, batch, protect=None) -> np.ndarray:
    """Parametric location-scale ComBat, samples x genes."""
    X = np.asarray(X, dtype=np.float64)
    batch = pd.Series(batch).astype("category")
    n, g = X.shape
    batches = list(batch.cat.categories)
    if len(batches) < 2:
        return X.copy()

    prot = _one_hot(protect, drop_first=True) if protect is not None else np.zeros((n, 0))
    C_bio = np.hstack([np.ones((n, 1)), prot])
    coef_bio = _lstsq_coef(C_bio, X)
    fitted = C_bio @ coef_bio
    resid = X - fitted
    gene_sd = resid.std(axis=0, ddof=1)
    gene_sd = np.where(gene_sd < 1e-8, 1.0, gene_sd)
    Z = resid / gene_sd

    gamma_hat = np.zeros((len(batches), g))
    delta_hat = np.ones((len(batches), g))
    n_b = np.zeros(len(batches))
    for i, b in enumerate(batches):
        mask = batch.values == b
        n_b[i] = float(mask.sum())
        if n_b[i] < 2:
            continue
        Zb = Z[mask]
        gamma_hat[i] = Zb.mean(axis=0)
        delta_hat[i] = np.clip(Zb.var(axis=0, ddof=1), 1e-6, None)

    gamma_bar = gamma_hat.mean(axis=0)
    tau2 = np.clip(gamma_hat.var(axis=0, ddof=0), 1e-6, None)
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    for i in range(len(batches)):
        w = n_b[i] / (n_b[i] + 1.0 / tau2)
        gamma_star[i] = w * gamma_hat[i] + (1.0 - w) * gamma_bar
        a = (n_b[i] / 2.0) + 1.0
        bpri = (n_b[i] * delta_hat[i] / 2.0) + 1.0
        delta_star[i] = np.clip(bpri / a, 1e-4, None)

    Z_adj = np.empty_like(Z)
    for i, b in enumerate(batches):
        mask = batch.values == b
        Z_adj[mask] = (Z[mask] - gamma_star[i]) / np.sqrt(delta_star[i])
    return Z_adj * gene_sd + fitted


def sva_correct(X: np.ndarray, protect, n_sv: int = 10) -> np.ndarray:
    """SVA-like: SVD on biology residual, regress surrogate variables out."""
    X = np.asarray(X, dtype=np.float64)
    C = np.hstack(
        [
            np.ones((X.shape[0], 1)),
            _one_hot(protect, drop_first=True),
        ]
    )
    coef = _lstsq_coef(C, X)
    resid = X - C @ coef
    k = int(min(max(n_sv, 1), X.shape[0] - 2, X.shape[1] - 1, 20))
    sv = TruncatedSVD(n_components=k, random_state=0).fit_transform(resid)
    C2 = np.hstack([C, sv])
    coef2 = _lstsq_coef(C2, X)
    return X - sv @ coef2[C.shape[1] :]


def harmony_correct(X: np.ndarray, batch, n_pcs: int = 30, n_iter: int = 8) -> np.ndarray:
    """Harmony-style iterative batch mixing in PCA, inverse-projected to genes.

    Uses sklearn only (harmonypy 2.x failed to build on this machine).
    """
    from sklearn.cluster import MiniBatchKMeans

    X = np.asarray(X, dtype=np.float64)
    batch = np.asarray(batch)
    n_pcs = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_pcs, random_state=0)
    z = pca.fit_transform(X)
    n_plat = len(np.unique(batch))
    n_clusters = int(min(max(n_plat * 3, 8), max(z.shape[0] // 15, n_plat + 1)))
    for it in range(n_iter):
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            batch_size=min(1024, z.shape[0]),
            n_init=3,
        )
        cl = km.fit_predict(z)
        for c in np.unique(cl):
            idx_c = np.where(cl == c)[0]
            if len(idx_c) < 4:
                continue
            mu_c = z[idx_c].mean(axis=0)
            for b in np.unique(batch):
                idx = idx_c[batch[idx_c] == b]
                if len(idx) < 2:
                    continue
                mu_cb = z[idx].mean(axis=0)
                z[idx] = z[idx] - 0.85 * (mu_cb - mu_c)
    return pca.inverse_transform(z)


def ruvg_correct(X: np.ndarray, n_factors: int = 5, n_controls: int = 1000) -> np.ndarray:
    """RUVg-like: least-variable genes as empirical negative controls."""
    X = np.asarray(X, dtype=np.float64)
    v = X.var(axis=0)
    n_controls = int(min(n_controls, max(X.shape[1] // 5, 50), X.shape[1] - 1))
    ctrl = np.argsort(v)[:n_controls]
    ctrl_x = X[:, ctrl]
    ctrl_x = ctrl_x - ctrl_x.mean(axis=0, keepdims=True)
    k = int(min(n_factors, ctrl_x.shape[0] - 1, ctrl_x.shape[1] - 1, 10))
    w = TruncatedSVD(n_components=k, random_state=0).fit_transform(ctrl_x)
    coef = _lstsq_coef(np.hstack([np.ones((X.shape[0], 1)), w]), X)
    return X - w @ coef[1:]


def _pca_pack(X: np.ndarray, n_pcs: int, seed: int = 0):
    X = np.asarray(X, dtype=np.float64)
    n_pcs = int(min(max(n_pcs, 8), X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_pcs, random_state=seed)
    z = pca.fit_transform(X).astype(np.float32)
    mu = z.mean(axis=0, keepdims=True)
    sd = np.clip(z.std(axis=0, keepdims=True), 1e-6, None)
    return pca, (z - mu) / sd, mu, sd


def _pca_unpack(pca, z_std, mu, sd) -> np.ndarray:
    z = np.asarray(z_std, dtype=np.float64) * sd + mu
    return pca.inverse_transform(z)


def respan_correct(X: np.ndarray, batch, n_pcs: int = 50, epochs: int = 120, seed: int = 0):
    """ResPAN-style residual WGAN in PCA space, mapped back to genes.

    Official ResPAN is scRNA-seq + scanpy; this follows the same residual
    generator + critic idea on bulk matrices.
    """
    import torch
    import torch.nn as nn

    from idfb.utils import get_device

    pca, z, mu, sd = _pca_pack(X, n_pcs, seed)
    codes = pd.Series(batch).astype("category").cat.codes.to_numpy()
    if int(codes.max()) < 1:
        return np.asarray(X, dtype=np.float64).copy()
    device = get_device()
    torch.manual_seed(seed)
    d = z.shape[1]
    gen = nn.Sequential(
        nn.Linear(d, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, d),
    ).to(device)
    disc = nn.Sequential(
        nn.Linear(d, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3, betas=(0.5, 0.9))
    z_t = torch.tensor(z, device=device)
    codes_t = torch.tensor(codes, device=device)
    ref = int(np.bincount(codes).argmax())
    z_ref = z_t[codes_t == ref]
    q_idx = torch.where(codes_t != ref)[0]
    if int(q_idx.numel()) == 0:
        return np.asarray(X, dtype=np.float64).copy()

    def residual(x):
        return x + gen(x)

    for _ in range(int(epochs)):
        q = z_t[q_idx]
        fake = residual(q).detach()
        d_loss = disc(fake).mean() - disc(z_ref).mean()
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        fake = residual(q)
        g_loss = -disc(fake).mean() + 0.05 * ((fake - q) ** 2).mean()
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    with torch.no_grad():
        z_out = z_t.clone()
        z_out[q_idx] = residual(z_t[q_idx])
    return _pca_unpack(pca, z_out.cpu().numpy(), mu, sd)


def bermad_correct(X: np.ndarray, batch, n_pcs: int = 50, epochs: int = 120, seed: int = 0):
    """BERMAD-style adaptation AE: reconstruct PCA scores + MMD across platforms."""
    import torch
    import torch.nn as nn

    from idfb.utils import get_device

    pca, z, mu, sd = _pca_pack(X, n_pcs, seed)
    codes = pd.Series(batch).astype("category").cat.codes.to_numpy()
    n_plat = int(codes.max()) + 1
    if n_plat < 2:
        return np.asarray(X, dtype=np.float64).copy()
    device = get_device()
    torch.manual_seed(seed)
    d = z.shape[1]
    hid, lat = 64, 20
    enc = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, lat)).to(device)
    dec = nn.Sequential(nn.Linear(lat, hid), nn.ReLU(), nn.Linear(hid, d)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    z_t = torch.tensor(z, device=device)
    codes_t = torch.tensor(codes, device=device)

    def rbf_mmd(a, b, bw=1.0):
        aa = (a * a).sum(1, keepdim=True)
        bb = (b * b).sum(1, keepdim=True)
        k_aa = torch.exp(-(aa + aa.T - 2 * a @ a.T) / (2 * bw * bw))
        k_bb = torch.exp(-(bb + bb.T - 2 * b @ b.T) / (2 * bw * bw))
        k_ab = torch.exp(-(aa + bb.T - 2 * a @ b.T) / (2 * bw * bw))
        return k_aa.mean() + k_bb.mean() - 2 * k_ab.mean()

    for _ in range(int(epochs)):
        h = enc(z_t)
        recon = dec(h)
        loss = ((recon - z_t) ** 2).mean()
        mmd = z_t.new_zeros(())
        n_pairs = 0
        for i in range(n_plat):
            for j in range(i + 1, n_plat):
                hi = h[codes_t == i]
                hj = h[codes_t == j]
                if int(hi.shape[0]) < 8 or int(hj.shape[0]) < 8:
                    continue
                n_i = min(256, int(hi.shape[0]))
                n_j = min(256, int(hj.shape[0]))
                mmd = mmd + rbf_mmd(hi[:n_i], hj[:n_j])
                n_pairs += 1
        if n_pairs:
            loss = loss + 0.2 * mmd / n_pairs
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        z_out = dec(enc(z_t)).cpu().numpy()
    return _pca_unpack(pca, z_out, mu, sd)


def scvi_correct(
    X: np.ndarray,
    batch,
    n_pcs: int = 50,
    n_latent: int = 16,
    epochs: int = 100,
    seed: int = 0,
    kl_weight: float = 0.5,
):
    """Bulk-adapted scVI-style VAE: decode with a shared reference batch.

    Operates in PCA space (gene matrices are dense bulk, not counts). Platform
    one-hot conditions the decoder; correction re-decodes every sample with the
    majority platform so batch is stripped. No biology labels are used.
    """
    import torch
    import torch.nn as nn

    from idfb.utils import get_device

    pca, z, mu, sd = _pca_pack(X, n_pcs, seed)
    codes = pd.Series(batch).astype("category").cat.codes.to_numpy()
    n_plat = int(codes.max()) + 1
    if n_plat < 2:
        return np.asarray(X, dtype=np.float64).copy()

    device = get_device()
    torch.manual_seed(seed)
    d = z.shape[1]
    hid = 64
    enc_mu = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, n_latent)).to(device)
    enc_logvar = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, n_latent)).to(device)
    dec = nn.Sequential(
        nn.Linear(n_latent + n_plat, hid),
        nn.ReLU(),
        nn.Linear(hid, hid),
        nn.ReLU(),
        nn.Linear(hid, d),
    ).to(device)
    params = list(enc_mu.parameters()) + list(enc_logvar.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    z_t = torch.tensor(z, device=device)
    codes_t = torch.tensor(codes, device=device, dtype=torch.long)
    batch_oh = torch.nn.functional.one_hot(codes_t, num_classes=n_plat).float()
    ref = int(np.bincount(codes).argmax())
    ref_oh = torch.zeros_like(batch_oh)
    ref_oh[:, ref] = 1.0

    def reparam(m, lv):
        return m + torch.randn_like(m) * torch.exp(0.5 * lv)

    n = z_t.shape[0]
    bs = min(512, n)
    for ep in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = z_t[idx]
            bb = batch_oh[idx]
            m = enc_mu(xb)
            lv = enc_logvar(xb).clamp(-8, 4)
            h = reparam(m, lv)
            recon = dec(torch.cat([h, bb], dim=1))
            recon_loss = ((recon - xb) ** 2).mean()
            kl = -0.5 * (1 + lv - m.pow(2) - lv.exp()).mean()
            # mild KL warmup; keep latent informative but not over-collapsed
            beta = kl_weight * min(1.0, (ep + 1) / 20.0)
            loss = recon_loss + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        m = enc_mu(z_t)
        # deterministic mean encoding; decode under reference platform
        z_out = dec(torch.cat([m, ref_oh], dim=1)).cpu().numpy()
    return _pca_unpack(pca, z_out, mu, sd)


def apply_method(name: str, X: np.ndarray, batch, y) -> np.ndarray:
    name = name.lower()
    if name in ("uncorrected", "raw"):
        return uncorrected(X)
    if name == "limma":
        return limma_remove_batch(X, batch, protect=y)
    if name == "combat":
        return combat(X, batch, protect=y)
    if name == "sva":
        return sva_correct(X, protect=y)
    if name == "ruvg":
        return ruvg_correct(X)
    if name == "harmony":
        return harmony_correct(X, batch)
    if name in ("scvi", "scvi-lite", "scvi_bulk"):
        return scvi_correct(X, batch)
    if name == "bermad":
        return bermad_correct(X, batch)
    if name == "respan":
        return respan_correct(X, batch)
    raise ValueError(f"Unknown method: {name}")
