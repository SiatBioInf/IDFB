import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from idfb.config import (
    ADV_ON_CORRECTED,
    BATCH_SIZE,
    DISC_LR_SCALE,
    DISC_UPDATE_EVERY,
    GAN_WARMUP_EPOCHS,
    GENE_WEIGHT_POWER,
    GRAD_CLIP_NORM,
    GRL_LAMBDA,
    HUBER_DELTA,
    INPUT_DIM,
    KL_BETA_MAX,
    KL_MAX,
    KL_WARMUP_EPOCHS,
    LAMBDA_ADV,
    LAMBDA_DESIGN_CENTER,
    LAMBDA_DESIGN_REF,
    LAMBDA_DESIGN_Z,
    LAMBDA_FID_HINGE,
    LAMBDA_LATENT_ADV,
    LAMBDA_MMD,
    LAMBDA_PLAT_MEAN,
    LAMBDA_RECON_CORR,
    LAMBDA_Z_WITHIN,
    LAMBDA_REF_CORR,
    LAMBDA_REF_MSE,
    LAMBDA_REF_SRC,
    LAMBDA_REF_VAR,
    FID_HINGE_TARGET,
    LATENT_ADV_WARMUP_EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
    MAX_GEN_LOSS,
    MODEL_DIR,
    N_EPOCHS,
    N_GPLS,
    PLATFORMS,
    RECON_LOSS,
    REFERENCE_PLATFORM_ID,
    SEED,
    SELECT_BY_FIDELITY,
    SELECT_LEAK_PENALTY,
    SELECT_MAX_VAL_RATIO,
    SELECT_MIN_CORR,
    SPLIT_BY_DESIGN,
    TRAIN_USE_ADVERSARIAL,
    USE_DESIGN_PAIR_LOSS,
    USE_LATENT_ADV,
    USE_REF_DECODE_LOSS,
    WEIGHT_DECAY,
)
from idfb.dataset import build_dataloaders
from idfb.models import Discriminator, LatentPlatformHead, VAE, grad_reverse
from idfb.utils import ensure_dir, get_device, set_seed


def batch_pearson_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    num = (a_c * b_c).sum(dim=1)
    den = torch.sqrt((a_c.pow(2).sum(dim=1) + eps) * (b_c.pow(2).sum(dim=1) + eps))
    corr = num / den
    return 1.0 - corr.mean()


def batch_pearson_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    num = (a_c * b_c).sum(dim=1)
    den = torch.sqrt((a_c.pow(2).sum(dim=1) + eps) * (b_c.pow(2).sum(dim=1) + eps))
    return (num / den).mean()


def gene_recon_weights(targets: torch.Tensor, power: float = GENE_WEIGHT_POWER) -> torch.Tensor:
    """Down-weight high-mean genes so recon is less dominated by them."""
    mean_abs = targets.detach().abs().mean(dim=0).clamp_min(1e-6)
    w = mean_abs.pow(-float(power))
    return w / w.mean()


def reconstruction_loss(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    w = gene_recon_weights(inputs)
    if RECON_LOSS == "huber":
        per = F.smooth_l1_loss(outputs, inputs, beta=float(HUBER_DELTA), reduction="none")
    else:
        per = (outputs - inputs).pow(2)
    recon = (per * w.unsqueeze(0)).mean()
    if LAMBDA_RECON_CORR > 0:
        recon = recon + float(LAMBDA_RECON_CORR) * batch_pearson_loss(outputs, inputs)
    return recon


def gene_huber(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(a, b, beta=float(HUBER_DELTA), reduction="mean")


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """RBF MMD^2 between two sets of vectors (used on L2-normalized z_clean)."""
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_zeros(())
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx = torch.exp(-gamma * xx).mean()
    kyy = torch.exp(-gamma * yy).mean()
    kxy = torch.exp(-gamma * xy).mean()
    return kxx + kyy - 2.0 * kxy


def platform_mmd(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    plats = p.unique()
    acc = z.new_zeros(())
    n = 0
    z_n = F.normalize(z, dim=1)
    for i, a in enumerate(plats):
        za = z_n[p == a]
        if za.size(0) < 4:
            continue
        for b in plats[i + 1 :]:
            zb = z_n[p == b]
            if zb.size(0) < 4:
                continue
            acc = acc + rbf_mmd(za, zb, sigma=0.5)
            n += 1
    if n == 0:
        return z.new_zeros(())
    return acc / n


def get_vae_loss(inputs, outputs, z_mean, z_std, kl_beta_w=1.0):
    recon = reconstruction_loss(outputs, inputs)
    logvar = 2.0 * torch.log(z_std.clamp(1e-4, 5.0))
    kl_loss = -0.5 * torch.mean(1.0 + logvar - z_mean.pow(2) - logvar.exp())
    kl_loss = kl_loss.clamp(max=float(KL_MAX))
    return recon + float(kl_beta_w) * kl_loss, recon, kl_loss


def variance_match_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vx = x.var(unbiased=False) + eps
    vy = y.var(unbiased=False) + eps
    ratio = vy / vx
    return F.relu(1.0 - ratio) + (ratio - 1.0).abs() * 0.25


def soft_cross_entropy(logits, soft_targets):
    log_prob = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()


def uniform_targets(batch_size, n_classes, device):
    return torch.full((batch_size, n_classes), 1.0 / n_classes, device=device)


def stop_training_minimize(val_scores, patience=15, min_delta=1e-4):
    if len(val_scores) < patience + 1:
        return False
    best_before = min(val_scores[:-patience])
    best_recent = min(val_scores[-patience:])
    improved = best_before - best_recent > min_delta
    return not improved


def stop_training_maximize(val_scores, patience=15, min_delta=1e-4):
    if len(val_scores) < patience + 1:
        return False
    best_before = max(val_scores[:-patience])
    best_recent = max(val_scores[-patience:])
    improved = best_recent - best_before > min_delta
    return not improved


def unpack_batch(batch, device):
    if len(batch) == 2:
        x, p = batch
        return x.to(device), p.long().to(device), None, None, None
    x, p, design_id, x_pair, p_pair = batch
    return (
        x.to(device),
        p.long().to(device),
        design_id.long().to(device),
        x_pair.to(device),
        p_pair.long().to(device),
    )


def latent_adv_lambda(epoch: int) -> float:
    if not USE_LATENT_ADV:
        return 0.0
    if epoch <= LATENT_ADV_WARMUP_EPOCHS:
        return LAMBDA_LATENT_ADV * float(epoch) / float(max(LATENT_ADV_WARMUP_EPOCHS, 1))
    return LAMBDA_LATENT_ADV


def kl_beta(epoch: int) -> float:
    if epoch <= KL_WARMUP_EPOCHS:
        return float(KL_BETA_MAX) * float(epoch) / float(max(KL_WARMUP_EPOCHS, 1))
    return float(KL_BETA_MAX)


def gan_adv_lambda(epoch: int, use_adversarial: bool) -> float:
    """Expression-GAN weight: 0 during warmup, then ramp to LAMBDA_ADV."""
    if not use_adversarial:
        return 0.0
    if epoch <= GAN_WARMUP_EPOCHS:
        return 0.0
    # short ramp after warmup (same length as latent warmup)
    ramp = max(LATENT_ADV_WARMUP_EPOCHS, 1)
    t = epoch - GAN_WARMUP_EPOCHS
    if t < ramp:
        return LAMBDA_ADV * float(t) / float(ramp)
    return LAMBDA_ADV


def compute_generator_loss(
    vae,
    x,
    p,
    x_pair=None,
    p_pair=None,
    design_id=None,
    n_gpls=N_GPLS,
    use_adversarial=False,
    disc=None,
    latent_head=None,
    latent_adv_w=0.0,
    expr_adv_w=0.0,
    use_ref_decode=USE_REF_DECODE_LOSS,
    use_design_pair=USE_DESIGN_PAIR_LOSS,
    reference_platform_id=REFERENCE_PLATFORM_ID,
    adv_on_corrected=ADV_ON_CORRECTED,
    kl_beta_w=1.0,
):
    """VAE fidelity + pair/ref + latent GRL + expression GAN on corrected x̂."""
    recon, loc, std = vae(x, p)
    vae_loss, recon_term, kl_term = get_vae_loss(x, recon, loc, std, kl_beta_w=kl_beta_w)
    loss = vae_loss
    parts = {
        "vae": float(vae_loss.detach()),
        "recon": float(recon_term.detach()),
        "kl": float(kl_term.detach()),
    }

    p_ref = torch.full_like(p, int(reference_platform_id))
    recon_ref = None

    if use_ref_decode:
        loc_clean = vae.clean_latent(loc, p)
        recon_ref = vae.decode(loc_clean, p_ref)
        # Do NOT pull corrected toward raw x (that copies platform fingerprints).
        # Anchor corrected to source-recon structure/variance instead.
        ref_src = gene_huber(recon_ref, recon.detach())
        ref_var = variance_match_loss(recon.detach(), recon_ref)
        ref_term = LAMBDA_REF_SRC * ref_src + LAMBDA_REF_VAR * ref_var
        if LAMBDA_REF_MSE > 0:
            ref_mse = gene_huber(recon_ref, x)
            ref_term = ref_term + LAMBDA_REF_MSE * ref_mse
            parts["ref_mse"] = float(ref_mse.detach())
        if LAMBDA_REF_CORR > 0:
            ref_corr = batch_pearson_loss(recon_ref, x)
            ref_term = ref_term + LAMBDA_REF_CORR * ref_corr
            parts["ref_corr"] = float(ref_corr.detach())
        loss = loss + ref_term
        parts["ref_src"] = float(ref_src.detach())
        parts["ref_var"] = float(ref_var.detach())

        if LAMBDA_FID_HINGE > 0:
            # Maximize corr(corrected, x) up to target via hinge (keeps biology).
            corr_cx = batch_pearson_mean(recon_ref, x)
            hinge = F.relu(FID_HINGE_TARGET - corr_cx)
            loss = loss + LAMBDA_FID_HINGE * hinge
            parts["fid_hinge"] = float(hinge.detach())
            parts["corr_cx"] = float(corr_cx.detach())

    if use_design_pair and x_pair is not None and p_pair is not None:
        loc_pair, std_pair = vae.encode(x_pair)
        loc_clean = vae.clean_latent(loc, p) if recon_ref is not None else vae.clean_latent(loc, p)
        loc_pair_clean = vae.clean_latent(loc_pair, p_pair)
        pair_z = gene_huber(loc_clean, loc_pair_clean)
        if recon_ref is None:
            recon_ref = vae.decode(loc_clean, p_ref)
        recon_ref_pair = vae.decode(loc_pair_clean, p_ref)
        pair_ref = gene_huber(recon_ref, recon_ref_pair)
        pair_term = LAMBDA_DESIGN_Z * pair_z + LAMBDA_DESIGN_REF * pair_ref
        loss = loss + pair_term
        parts["pair_z"] = float(pair_z.detach())
        parts["pair_ref"] = float(pair_ref.detach())
        pair_vae, _, _ = get_vae_loss(
            x_pair,
            vae.decode(loc_pair, p_pair),
            loc_pair,
            std_pair,
            kl_beta_w=kl_beta_w,
        )
        loss = loss + 0.10 * pair_vae

    # Same-design corrected vectors → batch centroid (cross-platform collapse).
    if (
        LAMBDA_DESIGN_CENTER > 0
        and design_id is not None
        and recon_ref is not None
    ):
        center_loss = torch.tensor(0.0, device=x.device)
        n_groups = 0
        for d in design_id.unique():
            mask = design_id == d
            if int(mask.sum()) < 2:
                continue
            group = recon_ref[mask]
            center = group.mean(dim=0, keepdim=True).detach()
            center_loss = center_loss + gene_huber(group, center.expand_as(group))
            n_groups += 1
        if n_groups > 0:
            center_loss = center_loss / n_groups
            loss = loss + LAMBDA_DESIGN_CENTER * center_loss
            parts["design_center"] = float(center_loss.detach())

    # Collapse cleaned latents within design (makes per-sample inference platform-free).
    if LAMBDA_Z_WITHIN > 0 and design_id is not None:
        loc_c = vae.clean_latent(loc, p)
        z_within = torch.tensor(0.0, device=x.device)
        n_groups = 0
        for d in design_id.unique():
            mask = design_id == d
            if int(mask.sum()) < 2:
                continue
            g = loc_c[mask]
            z_within = z_within + g.var(dim=0, unbiased=False).mean()
            n_groups += 1
        if n_groups > 0:
            z_within = z_within / n_groups
            loss = loss + LAMBDA_Z_WITHIN * z_within
            parts["z_within"] = float(z_within.detach())

    # Match per-platform means of z_clean → improves cross-platform neighbor mixing.
    if LAMBDA_PLAT_MEAN > 0:
        loc_c = vae.clean_latent(loc, p)
        global_mu = loc_c.mean(dim=0)
        plat_term = torch.tensor(0.0, device=x.device)
        n_plat = 0
        for plat in p.unique():
            mask = p == plat
            if int(mask.sum()) < 2:
                continue
            plat_term = plat_term + F.mse_loss(loc_c[mask].mean(dim=0), global_mu)
            n_plat += 1
        if n_plat > 0:
            plat_term = plat_term / n_plat
            loss = loss + LAMBDA_PLAT_MEAN * plat_term
            parts["plat_mean"] = float(plat_term.detach())

    if LAMBDA_MMD > 0:
        loc_c = vae.clean_latent(loc, p)
        mmd = platform_mmd(loc_c, p)
        loss = loss + float(LAMBDA_MMD) * mmd
        parts["mmd"] = float(mmd.detach())

    if latent_head is not None and latent_adv_w > 0:
        uniform = uniform_targets(x.size(0), n_gpls, x.device)
        loc_c = vae.clean_latent(loc, p)
        lat_logits = latent_head(loc_c, reverse=True, lambd=float(GRL_LAMBDA))
        lat_adv = soft_cross_entropy(lat_logits, uniform)
        loss = loss + latent_adv_w * lat_adv
        parts["lat_adv"] = float(lat_adv.detach())

    # Expression GAN: fool D on corrected output toward uniform (GRL + soft-CE).
    if use_adversarial and disc is not None and expr_adv_w > 0:
        if recon_ref is None:
            recon_ref = vae.decode(vae.clean_latent(loc, p), p_ref)
        target = recon_ref if adv_on_corrected else recon
        logits = disc(grad_reverse(target, lambd=float(GRL_LAMBDA)))
        adv_loss = soft_cross_entropy(
            logits, uniform_targets(x.size(0), n_gpls, x.device)
        )
        loss = loss + expr_adv_w * adv_loss
        parts["adv"] = float(adv_loss.detach())

    parts["total"] = float(loss.detach())
    return loss, parts


@torch.no_grad()
def fidelity_select_score(
    dataloader,
    vae,
    device="cpu",
    reference_platform_id=REFERENCE_PLATFORM_ID,
    max_batches=16,
):
    """Higher is better: correlation of reconstructions to raw x + variance retention."""
    vae.eval()
    src_corrs = []
    cor_corrs = []
    src_vars = []
    cor_vars = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x, p, _, _, _ = unpack_batch(batch, device)
        loc = vae.encode_mu(x)
        recon = vae.decode(loc, p)
        p_ref = torch.full_like(p, int(reference_platform_id))
        recon_ref = vae.decode(vae.clean_latent(loc, p), p_ref)
        src_corrs.append(float(batch_pearson_mean(recon, x)))
        cor_corrs.append(float(batch_pearson_mean(recon_ref, x)))
        vx = float(x.var(unbiased=False).clamp_min(1e-8))
        src_vars.append(float(recon.var(unbiased=False) / vx))
        cor_vars.append(float(recon_ref.var(unbiased=False) / vx))

    if not src_corrs:
        return 0.0, 0.0, 0.0
    src_c = sum(src_corrs) / len(src_corrs)
    cor_c = sum(cor_corrs) / len(cor_corrs)
    src_v = min(sum(src_vars) / len(src_vars), 1.5)
    cor_v = min(sum(cor_vars) / len(cor_vars), 1.5)
    score = 0.45 * src_c + 0.45 * cor_c + 0.10 * min(cor_v, 1.2)
    return score, cor_c, src_c


@torch.no_grad()
def corrected_platform_leak(
    dataloader,
    vae,
    disc,
    device="cpu",
    reference_platform_id=REFERENCE_PLATFORM_ID,
    max_batches=16,
) -> float:
    """Disc balanced-ish accuracy on corrected outputs (higher = more leak)."""
    if disc is None:
        return 0.0
    vae.eval()
    disc.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x, p, _, _, _ = unpack_batch(batch, device)
        loc = vae.encode_mu(x)
        p_ref = torch.full_like(p, int(reference_platform_id))
        corr = vae.decode(vae.clean_latent(loc, p), p_ref)
        pred = disc(corr).argmax(dim=1)
        correct += int((pred == p).sum().item())
        total += int(p.numel())
    if total == 0:
        return 0.0
    return correct / total


def selection_monitor(
    dataloader,
    vae,
    disc=None,
    device="cpu",
    reference_platform_id=REFERENCE_PLATFORM_ID,
    n_gpls=N_GPLS,
    leak_penalty=SELECT_LEAK_PENALTY,
):
    """Higher better: fidelity minus excess platform leak above chance."""
    fid, cor_c, src_c = fidelity_select_score(
        dataloader, vae, device=device, reference_platform_id=reference_platform_id
    )
    leak = corrected_platform_leak(
        dataloader,
        vae,
        disc,
        device=device,
        reference_platform_id=reference_platform_id,
    )
    penalty = float(leak_penalty) * max(0.0, leak - 1.0 / max(n_gpls, 1))
    if src_c < 0.50:
        score = src_c - 0.50
    else:
        score = 0.55 * src_c + 0.45 * cor_c - penalty
    return score, fid, leak, src_c, cor_c


def validate(
    dataloader,
    vae,
    disc=None,
    latent_head=None,
    device="cpu",
    n_gpls=N_GPLS,
    use_adversarial=True,
    use_ref_decode=USE_REF_DECODE_LOSS,
    use_design_pair=USE_DESIGN_PAIR_LOSS,
    latent_adv_w=0.0,
    expr_adv_w=0.0,
    reference_platform_id=REFERENCE_PLATFORM_ID,
    kl_beta_w=1.0,
):
    vae.eval()
    if disc is not None:
        disc.eval()
    if latent_head is not None:
        latent_head.eval()
    criterion = nn.CrossEntropyLoss()
    disc_running_loss = 0.0
    gen_running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x, p, design_id, x_pair, p_pair = unpack_batch(batch, device)
            gen_loss, _ = compute_generator_loss(
                vae,
                x,
                p,
                x_pair=x_pair,
                p_pair=p_pair,
                design_id=design_id,
                n_gpls=n_gpls,
                use_adversarial=use_adversarial,
                disc=disc,
                latent_head=latent_head,
                latent_adv_w=latent_adv_w,
                expr_adv_w=expr_adv_w,
                use_ref_decode=use_ref_decode,
                use_design_pair=use_design_pair and x_pair is not None,
                reference_platform_id=reference_platform_id,
                kl_beta_w=kl_beta_w,
            )
            gen_running_loss += gen_loss.item()

            if use_adversarial and disc is not None and expr_adv_w > 0:
                disc_loss = criterion(disc(x), p)
                disc_running_loss += disc_loss.item()

    n_batches = max(len(dataloader), 1)
    disc_val = disc_running_loss / n_batches if use_adversarial else 0.0
    gen_val = gen_running_loss / n_batches
    return disc_val, gen_val


def save_checkpoint(
    vae,
    disc,
    input_dim,
    latent_dim,
    n_gpls,
    model_dir,
    tag,
    meta=None,
    latent_head=None,
    platforms=None,
):
    ensure_dir(model_dir)
    plat_list = list(platforms) if platforms is not None else list(PLATFORMS)
    payload_vae = {
        "model_state_dict": vae.state_dict(),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "n_gpls": n_gpls,
        "platforms": plat_list,
    }
    if meta:
        payload_vae.update(meta)
    if latent_head is not None:
        payload_vae["latent_head_state_dict"] = latent_head.state_dict()

    vae_path = Path(model_dir) / f"vae_model{tag}.pth"
    torch.save(payload_vae, vae_path)

    disc_path = None
    if disc is not None:
        payload_disc = {
            "model_state_dict": disc.state_dict(),
            "input_dim": input_dim,
            "n_gpls": n_gpls,
            "platforms": plat_list,
        }
        if meta:
            payload_disc.update(meta)
        disc_path = Path(model_dir) / f"discriminator_model{tag}.pth"
        torch.save(payload_disc, disc_path)
    return vae_path, disc_path


def save_models(
    vae,
    disc,
    input_dim,
    latent_dim,
    n_gpls,
    model_dir=MODEL_DIR,
    meta=None,
    latent_head=None,
    platforms=None,
):
    vae_path, disc_path = save_checkpoint(
        vae,
        disc,
        input_dim,
        latent_dim,
        n_gpls,
        model_dir,
        tag="",
        meta=meta,
        latent_head=latent_head,
        platforms=platforms,
    )
    print(f"Models saved to {model_dir}")
    return vae_path, disc_path


def fit(
    train_loader,
    val_loader,
    vae,
    disc,
    n_epochs,
    input_dim,
    latent_dim,
    lr=LEARNING_RATE,
    early_stopping=True,
    patience=15,
    device="cpu",
    n_gpls=N_GPLS,
    model_dir=MODEL_DIR,
    use_adversarial=True,
    use_ref_decode=USE_REF_DECODE_LOSS,
    use_design_pair=USE_DESIGN_PAIR_LOSS,
    use_latent_adv=USE_LATENT_ADV,
    select_by_fidelity=SELECT_BY_FIDELITY,
    reference_platform_id=REFERENCE_PLATFORM_ID,
    platforms=None,
):
    criterion = nn.CrossEntropyLoss()
    vae = vae.to(device)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=float(WEIGHT_DECAY))
    disc_opt = None
    if use_adversarial:
        if disc is None:
            raise ValueError("Discriminator required when use_adversarial=True")
        disc = disc.to(device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=lr * float(DISC_LR_SCALE))

    latent_head = None
    latent_opt = None
    if use_latent_adv:
        latent_head = LatentPlatformHead(latent_dim, n_gpls).to(device)
        latent_opt = torch.optim.Adam(latent_head.parameters(), lr=lr * float(DISC_LR_SCALE))

    val_monitor = []
    best_monitor = float("-inf") if select_by_fidelity else float("inf")
    best_epoch = -1
    best_state = None
    mode = "gan" if use_adversarial else "ref+pair+latadv"
    plat_list = list(platforms) if platforms is not None else list(PLATFORMS)
    global_step = 0
    adv_disabled = False
    stable_mode = False
    freeze_kl = float(KL_BETA_MAX)
    n_collapse = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_opt, T_max=max(int(n_epochs), 1), eta_min=1e-5
    )

    for epoch in range(n_epochs):
        epoch_1based = epoch + 1
        if stable_mode:
            lat_w = 0.0
            expr_w = 0.0
            kl_w = float(freeze_kl)
        else:
            lat_w = latent_adv_lambda(epoch_1based) if use_latent_adv else 0.0
            expr_w = 0.0 if adv_disabled else gan_adv_lambda(epoch_1based, use_adversarial)
            kl_w = kl_beta(epoch_1based)

        vae.train()
        if use_adversarial:
            disc.train()
        if latent_head is not None:
            latent_head.train()

        disc_running_loss = 0.0
        gen_running_loss = 0.0
        lat_running_loss = 0.0
        n_gen_ok = 0
        parts_sum = {}

        for batch in train_loader:
            x, p, design_id, x_pair, p_pair = unpack_batch(batch, device)
            global_step += 1

            # Do not pre-train D to zero loss before G starts (that caused collapse).
            if (
                use_adversarial
                and expr_w > 0
                and (global_step % max(int(DISC_UPDATE_EVERY), 1) == 0)
            ):
                disc_opt.zero_grad()
                disc_loss = criterion(disc(x), p)
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=GRAD_CLIP_NORM)
                disc_opt.step()
                disc_running_loss += disc_loss.item()

            if latent_head is not None and lat_w > 0 and (
                global_step % max(int(DISC_UPDATE_EVERY), 1) == 0
            ):
                latent_opt.zero_grad()
                with torch.no_grad():
                    loc = vae.encode_mu(x)
                    loc_c = vae.clean_latent(loc, p)
                lat_loss = criterion(latent_head(loc_c.detach()), p)
                lat_loss.backward()
                torch.nn.utils.clip_grad_norm_(latent_head.parameters(), max_norm=GRAD_CLIP_NORM)
                latent_opt.step()
                lat_running_loss += lat_loss.item()

            vae_opt.zero_grad()
            gen_loss, parts = compute_generator_loss(
                vae,
                x,
                p,
                x_pair=x_pair,
                p_pair=p_pair,
                design_id=design_id,
                n_gpls=n_gpls,
                use_adversarial=use_adversarial,
                disc=disc,
                latent_head=latent_head,
                latent_adv_w=lat_w,
                expr_adv_w=expr_w,
                use_ref_decode=use_ref_decode,
                use_design_pair=use_design_pair and x_pair is not None,
                reference_platform_id=reference_platform_id,
                kl_beta_w=kl_w,
            )
            if not torch.isfinite(gen_loss) or float(gen_loss.detach()) > float(MAX_GEN_LOSS):
                vae_opt.zero_grad(set_to_none=True)
                continue
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=GRAD_CLIP_NORM)
            vae_opt.step()
            gen_running_loss += gen_loss.item()
            n_gen_ok += 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + float(v)

        n_batches = max(len(train_loader), 1)
        disc_epoch_loss = disc_running_loss / n_batches
        gen_epoch_loss = (
            gen_running_loss / n_gen_ok if n_gen_ok else float("nan")
        )
        lat_epoch_loss = lat_running_loss / n_batches
        skip_frac = 1.0 - (n_gen_ok / float(n_batches))
        parts_avg = {
            k: v / max(n_gen_ok, 1) for k, v in parts_sum.items()
        }
        if skip_frac > 0.5:
            for g in vae_opt.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-5)
            print(
                f"  skip={skip_frac:.0%} steps; lr -> {vae_opt.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        if (
            use_adversarial
            and expr_w > 0
            and not adv_disabled
            and skip_frac > 0.5
        ):
            adv_disabled = True
            print(
                f"Expr-GAN disabled after epoch {epoch_1based}: "
                f"unstable steps={skip_frac:.0%} (keep best fidelity ckpt)."
            )

        disc_val, gen_val = validate(
            val_loader,
            vae,
            disc=disc,
            latent_head=latent_head,
            device=device,
            n_gpls=n_gpls,
            use_adversarial=use_adversarial,
            use_ref_decode=use_ref_decode,
            use_design_pair=use_design_pair,
            latent_adv_w=lat_w,
            expr_adv_w=expr_w,
            reference_platform_id=reference_platform_id,
            kl_beta_w=kl_w,
        )
        src_c = 0.0
        cor_c = 0.0
        if select_by_fidelity:
            monitor, fid_score, leak, src_c, cor_c = selection_monitor(
                val_loader,
                vae,
                disc=disc if use_adversarial else None,
                device=device,
                reference_platform_id=reference_platform_id,
                n_gpls=n_gpls,
            )
        else:
            fid_score, cor_c, src_c = fidelity_select_score(
                val_loader,
                vae,
                device=device,
                reference_platform_id=reference_platform_id,
            )
            leak = 0.0
            monitor = gen_val

        collapsed = False
        healthy_val = None if best_state is None else float(best_state["gen_val"])
        if healthy_val is not None and gen_val > float(SELECT_MAX_VAL_RATIO) * healthy_val:
            collapsed = True
            print(
                f"  collapse: val={gen_val:.4f} > {SELECT_MAX_VAL_RATIO}x "
                f"healthy={healthy_val:.4f}; restore epoch {best_epoch}",
                flush=True,
            )
            vae.load_state_dict(best_state["vae"])
            if use_adversarial and best_state["disc"] is not None:
                disc.load_state_dict(best_state["disc"])
            if latent_head is not None and best_state["latent_head"] is not None:
                latent_head.load_state_dict(best_state["latent_head"])
            if best_state.get("vae_opt") is not None:
                vae_opt.load_state_dict(best_state["vae_opt"])
            if latent_opt is not None and best_state.get("latent_opt") is not None:
                latent_opt.load_state_dict(best_state["latent_opt"])
            if disc_opt is not None and best_state.get("disc_opt") is not None:
                disc_opt.load_state_dict(best_state["disc_opt"])
            for g in vae_opt.param_groups:
                g["lr"] = max(float(g["lr"]) * 0.4, 1e-5)
            adv_disabled = True
            stable_mode = True
            freeze_kl = min(float(kl_w), 0.02)
            n_collapse += 1
            val_monitor = []
            print(
                f"  stable_mode=on kl_w={freeze_kl:.3f} lr={vae_opt.param_groups[0]['lr']:.2e} "
                f"(lat/gan frozen, Adam restored)",
                flush=True,
            )
            gen_val = healthy_val
            monitor = best_monitor
            src_c = float("nan")
            cor_c = float("nan")
            if n_collapse >= 3:
                print("  too many collapses; stop and keep best checkpoint.", flush=True)
                break

        if not collapsed:
            val_monitor.append(monitor)

        if select_by_fidelity:
            recon_ok = True
            if best_state is not None:
                recon_ok = gen_val <= float(SELECT_MAX_VAL_RATIO) * float(
                    best_state["gen_val"]
                )
            improved = (
                (not collapsed)
                and recon_ok
                and src_c == src_c
                and monitor > best_monitor + 1e-6
            )
            if best_state is None and (not collapsed) and src_c == src_c:
                improved = True
        else:
            improved = (not collapsed) and monitor < best_monitor - 1e-6

        mark = ""
        if improved:
            best_monitor = monitor
            best_epoch = epoch_1based
            best_state = {
                "vae": {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()},
                "disc": (
                    {k: v.detach().cpu().clone() for k, v in disc.state_dict().items()}
                    if use_adversarial
                    else None
                ),
                "latent_head": (
                    {
                        k: v.detach().cpu().clone()
                        for k, v in latent_head.state_dict().items()
                    }
                    if latent_head is not None
                    else None
                ),
                "disc_val": disc_val,
                "gen_val": gen_val,
                "fid_score": fid_score,
                "leak": leak,
                "vae_opt": deepcopy(vae_opt.state_dict()),
                "latent_opt": (
                    deepcopy(latent_opt.state_dict()) if latent_opt is not None else None
                ),
                "disc_opt": (
                    deepcopy(disc_opt.state_dict()) if disc_opt is not None else None
                ),
            }
            meta = {
                "best_epoch": best_epoch,
                "best_val_disc": disc_val,
                "best_val_gen": gen_val,
                "best_fid_score": fid_score,
                "best_leak": leak,
                "train_mode": mode,
                "use_adversarial": use_adversarial,
                "adv_on_corrected": ADV_ON_CORRECTED,
                "use_ref_decode": use_ref_decode,
                "use_design_pair": use_design_pair,
                "use_latent_adv": use_latent_adv,
                "select_by_fidelity": select_by_fidelity,
            }
            save_checkpoint(
                vae,
                disc if use_adversarial else None,
                input_dim,
                latent_dim,
                n_gpls,
                model_dir,
                tag="_best",
                meta=meta,
                latent_head=latent_head,
                platforms=plat_list,
            )
            mark = " *best*"

        train_s = (
            f"{gen_epoch_loss:.4f}" if n_gen_ok else "skip"
        )
        print(
            f"Epoch {epoch_1based}/{n_epochs} [{mode}]: "
            f"train={train_s} | val={gen_val:.4f} | "
            f"src={src_c:.3f} cor={cor_c:.3f} | "
            f"recon={parts_avg.get('recon', 0):.4f} kl={parts_avg.get('kl', 0):.4f} "
            f"mmd={parts_avg.get('mmd', 0):.4f} | "
            f"fid={fid_score:.4f} | leak={leak:.3f} | mon={monitor:.4f} | "
            f"skip={skip_frac:.0%} | kl_w={kl_w:.3f} lat_w={lat_w:.3f} | "
            f"adv_w={expr_w:.3f} | lat_ce={lat_epoch_loss:.4f} | "
            f"disc={disc_epoch_loss:.4f}{mark}",
            flush=True,
        )
        scheduler.step()

        if early_stopping:
            should_stop = (
                stop_training_maximize(val_monitor, patience=patience)
                if select_by_fidelity
                else stop_training_minimize(val_monitor, patience=patience)
            )
            if should_stop:
                print(
                    f"Early stopping at epoch {epoch_1based} "
                    f"(patience={patience}, best_epoch={best_epoch})."
                )
                break

    if best_state is not None:
        vae.load_state_dict(best_state["vae"])
        if use_adversarial and best_state["disc"] is not None:
            disc.load_state_dict(best_state["disc"])
        if latent_head is not None and best_state["latent_head"] is not None:
            latent_head.load_state_dict(best_state["latent_head"])
        meta = {
            "best_epoch": best_epoch,
            "best_val_disc": best_state["disc_val"],
            "best_val_gen": best_state["gen_val"],
            "best_fid_score": best_state["fid_score"],
            "train_mode": mode,
            "use_adversarial": use_adversarial,
            "use_ref_decode": use_ref_decode,
            "use_design_pair": use_design_pair,
            "use_latent_adv": use_latent_adv,
            "select_by_fidelity": select_by_fidelity,
        }
        save_models(
            vae,
            disc if use_adversarial else None,
            input_dim,
            latent_dim,
            n_gpls,
            model_dir,
            meta=meta,
            latent_head=latent_head,
            platforms=plat_list,
        )
        print(
            f"Restored best checkpoint: epoch={best_epoch}, "
            f"fid={best_state['fid_score']:.4f}, val_gen={best_state['gen_val']:.4f}"
        )
    else:
        save_models(
            vae,
            disc if use_adversarial else None,
            input_dim,
            latent_dim,
            n_gpls,
            model_dir,
            meta={"train_mode": mode},
            latent_head=latent_head,
            platforms=plat_list,
        )

    return vae, disc, best_epoch, best_monitor


def run_train(
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    lr=LEARNING_RATE,
    latent_dim=LATENT_DIM,
    input_dim=INPUT_DIM,
    early_stopping=True,
    patience=15,
    seed=SEED,
    use_adversarial=TRAIN_USE_ADVERSARIAL,
    split_by_design=SPLIT_BY_DESIGN,
    use_ref_decode=USE_REF_DECODE_LOSS,
    use_design_pair=USE_DESIGN_PAIR_LOSS,
    use_latent_adv=USE_LATENT_ADV,
    select_by_fidelity=SELECT_BY_FIDELITY,
):
    set_seed(seed)
    with_partners = use_design_pair and split_by_design
    train_loader, test_loader = build_dataloaders(
        batch_size=batch_size,
        seed=seed,
        split_by_design=split_by_design,
        with_design_partners=with_partners,
    )
    n_gpls = N_GPLS
    assert n_gpls == len(PLATFORMS)

    sample = next(iter(train_loader))
    sample_x = sample[0]
    if sample_x.shape[1] != input_dim:
        raise ValueError(
            f"Feature dim mismatch: data={sample_x.shape[1]}, expected={input_dim}"
        )

    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    mode = "gan" if use_adversarial else "ref+pair+latadv"
    print(
        f"Platforms: {PLATFORMS}\n"
        f"Train mode: {mode}\n"
        f"use_adversarial={use_adversarial}\n"
        f"use_ref_decode={use_ref_decode}\n"
        f"use_design_pair={use_design_pair} (partners={with_partners})\n"
        f"use_latent_adv={use_latent_adv}\n"
        f"select_by_fidelity={select_by_fidelity}\n"
        f"Split by design: {split_by_design}\n"
        f"Train samples: {n_train}, Val samples: {n_test}, "
        f"batches/epoch: {len(train_loader)}"
    )

    vae = VAE(input_dim, latent_dim, n_gpls)
    disc = Discriminator(input_dim, n_gpls) if use_adversarial else None
    device = get_device()
    print(f"Device: {device}")

    vae, disc, best_epoch, best_monitor = fit(
        train_loader,
        test_loader,
        vae,
        disc,
        n_epochs=n_epochs,
        input_dim=input_dim,
        latent_dim=latent_dim,
        lr=lr,
        early_stopping=early_stopping,
        patience=patience,
        device=device,
        n_gpls=n_gpls,
        model_dir=MODEL_DIR,
        use_adversarial=use_adversarial,
        use_ref_decode=use_ref_decode,
        use_design_pair=use_design_pair,
        use_latent_adv=use_latent_adv,
        select_by_fidelity=select_by_fidelity,
    )
    print(f"Done. Best epoch={best_epoch}, best_monitor={best_monitor:.4f}, mode={mode}")
    return vae, disc


def main():
    parser = argparse.ArgumentParser(description="Train IDFB (baseline VAE or GAN-VAE)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--adversarial",
        action=argparse.BooleanOptionalAction,
        default=TRAIN_USE_ADVERSARIAL,
    )
    parser.add_argument(
        "--split-by-design",
        action=argparse.BooleanOptionalAction,
        default=SPLIT_BY_DESIGN,
    )
    parser.add_argument(
        "--ref-decode",
        action=argparse.BooleanOptionalAction,
        default=USE_REF_DECODE_LOSS,
    )
    parser.add_argument(
        "--design-pair",
        action=argparse.BooleanOptionalAction,
        default=USE_DESIGN_PAIR_LOSS,
    )
    parser.add_argument(
        "--latent-adv",
        action=argparse.BooleanOptionalAction,
        default=USE_LATENT_ADV,
    )
    args = parser.parse_args()
    run_train(
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        early_stopping=not args.no_early_stopping,
        seed=args.seed,
        use_adversarial=args.adversarial,
        split_by_design=args.split_by_design,
        use_ref_decode=args.ref_decode,
        use_design_pair=args.design_pair,
        use_latent_adv=args.latent_adv,
    )


if __name__ == "__main__":
    main()
