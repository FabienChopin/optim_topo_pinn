"""Réseaux : u avec conditions de Dirichlet fortes, rho = sigmoïde(N) dans ]0, 1[."""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP tanh, entrées [lo, hi]^d ramenées sur [-1, 1]^d."""

    def __init__(self, in_dim: int, width: int, depth: int, lo: float = 0.0, hi: float = 1.0):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.lo, self.hi = lo, hi
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0)

    @property
    def last(self) -> nn.Linear:
        return self.net[-1]


class StateNet(nn.Module):
    """u(x) = b(x) N(x), b nul sur le bord : u = 0 au bord par construction."""

    def __init__(self, mlp: MLP, boundary_factor):
        super().__init__()
        self.mlp = mlp
        self.boundary_factor = boundary_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_factor(x) * self.mlp(x)


class DensityNet(nn.Module):
    """rho(x) = sigmoïde(N(x)), initialisé presque uniforme à rho = v0."""

    def __init__(self, mlp: MLP, v0: float):
        super().__init__()
        self.mlp = mlp
        with torch.no_grad():
            mlp.last.weight.mul_(0.1)
            mlp.last.bias.fill_(math.log(v0 / (1.0 - v0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))
