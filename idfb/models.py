from typing import Optional, Tuple
import collections

import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        h_depth: int = 2,
        h_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = input_dim
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = nn.Linear(ptr_dim, latent_dim)
        self.std_lin = nn.Linear(ptr_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, eps: float = 1e-7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ptr = x
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr).clamp(-8.0, 8.0)
        std = F.softplus(self.std_lin(ptr)).clamp(max=2.0) + eps
        return loc, std


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        n_gpls: int,
        h_depth: int = 2,
        h_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_gpls = n_gpls
        self.h_depth = h_depth
        ptr_dim = latent_dim + self.n_gpls
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", nn.Dropout(p=dropout))
            ptr_dim = h_dim + self.n_gpls
        self.loc = nn.Linear(ptr_dim, input_dim)
        nn.init.xavier_uniform_(self.loc.weight, gain=0.01)
        nn.init.zeros_(self.loc.bias)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        p_one_hot = F.one_hot(p, num_classes=self.n_gpls).float()
        ptr = torch.cat([z, p_one_hot], dim=1)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
            ptr = torch.cat([ptr, p_one_hot], dim=1)
        return self.loc(ptr).clamp(-2.0, 2.0)


class VAE(nn.Module):
    def __init__(
        self, input_dim: int, latent_dim: int, n_gpls: int = 1
    ) -> None:
        super().__init__()
        self.n_gpls = n_gpls
        self.latent_dim = latent_dim
        self.encode = Encoder(input_dim, latent_dim)
        self.decode = Decoder(latent_dim, input_dim, n_gpls)
        # Explicit platform residual in latent space (known p at train/test).
        self.platform_bias = nn.Embedding(n_gpls, latent_dim)
        nn.init.zeros_(self.platform_bias.weight)

    def clean_latent(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Remove estimated platform component: z_clean = z - E[p]."""
        return z - self.platform_bias(p)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc, std = self.encode(x)
        z = D.Normal(loc, std).rsample()
        recon = self.decode(z, p)
        return recon, loc, std

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        loc, _ = self.encode(x)
        return loc

    def reconstruct(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        deterministic: bool = True,
        correct: bool = False,
        reference_platform_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, std = self.encode(x)
        z = loc if deterministic else D.Normal(loc, std).rsample()
        if correct:
            z = self.clean_latent(z, p)
            p_out = torch.full_like(p, int(reference_platform_id))
            recon = self.decode(z, p_out)
        else:
            recon = self.decode(z, p)
        return recon, loc


class Discriminator(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        n_gpls: int,
        h_depth: int = 2,
        h_dim: Optional[int] = 256,
        dropout: float = 0.2,
    ) -> None:
        self.n_gpls = n_gpls
        layers = collections.OrderedDict()
        ptr_dim = input_dim
        for layer in range(h_depth):
            layers[f"linear_{layer}"] = nn.Linear(ptr_dim, h_dim)
            layers[f"act_{layer}"] = nn.LeakyReLU(negative_slope=0.2)
            layers[f"dropout_{layer}"] = nn.Dropout(p=dropout)
            ptr_dim = h_dim
        layers["pred"] = nn.Linear(ptr_dim, n_gpls)
        super().__init__(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class LatentPlatformHead(nn.Module):
    """Small classifier on latent codes for adversarial platform removal."""

    def __init__(self, latent_dim: int, n_gpls: int, h_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(h_dim, n_gpls),
        )

    def forward(self, z: torch.Tensor, reverse: bool = False, lambd: float = 1.0):
        if reverse:
            z = grad_reverse(z, lambd)
        return self.net(z)
