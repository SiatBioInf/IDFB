from typing import Optional, Tuple
import collections
import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

class Encoder(nn.Module):
    r"""
    Abstract data encoder
    Parameters
        input_dim: Input dimensionality
        latent_dim: Latent dimensionality
        h_depth: Hidden layer depth
        h_dim: Hidden layer dimensionality
        dropout: Dropout rate
    """
    def __init__(
            self, input_dim: int, latent_dim: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
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
            self, x: torch.Tensor, EPS: float = 1e-7
        ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Encode data to sample latent distribution
        Parameters:
            x: Input data
        Returns:
            z: Sample latent distribution
        """       
        ptr = x
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return loc, std


class Decoder(nn.Module):
    def __init__(
            self, latent_dim: int, input_dim: int, n_gpls: int,
            h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.n_gpls = n_gpls
        self.h_depth = h_depth
        ptr_dim = latent_dim + self.n_gpls
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim + self.n_gpls
        self.loc = torch.nn.Linear(ptr_dim, input_dim)

    def forward(
            self, z: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        p_one_hot = F.one_hot(p, num_classes=self.n_gpls)
        ptr = torch.cat([z, p_one_hot], dim=1)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
            ptr = torch.cat([ptr, p_one_hot], dim=1)
        recon = self.loc(ptr)
        return recon


class VAE(nn.Module):
    """
    Parameters:
        input_dim: Input dimensionality
        latent_dim: Latent dimensionality
        n_gpls: The number of GPL
    """

    def __init__(self, input_dim: int, latent_dim: int, 
                 n_gpls: int = 1) -> None:
        super(VAE, self).__init__()
        self.encode = Encoder(input_dim, latent_dim)
        self.decode = Decoder(latent_dim, input_dim, n_gpls)

    def forward(
            self, x: torch.Tensor, p: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            x: Input data
            p: GPL index
        Returns:
            recon: Reconstructed data
            z_dist: the z-distribution of encoding
        """
        loc, std = self.encode(x)
        device = x.device
        z = D.Normal(loc, std).rsample().to(device)
        recon = self.decode(z, p)
        return recon, loc, std


class Discriminator(torch.nn.Sequential):

    def __init__(
            self, input_dim: int, n_gpls: int,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_gpls = n_gpls 
        od = collections.OrderedDict()
        ptr_dim = input_dim + self.n_gpls
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, 4) 
        super().__init__(od)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor: 
        if self.n_gpls:
            p_one_hot = F.one_hot(p, num_classes=self.n_gpls)
            x = torch.cat([x, p_one_hot], dim=1)
        return super().forward(x)

