"""Réseaux : u avec conditions de Dirichlet fortes, rho = sigmoïde(N) dans ]0, 1[ ; en forme mixte,
un réseau de plus pour le flux sigma = A grad u ; en forme de Ritz, un réseau de plus pour l'adjoint p."""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP tanh, entrées [lo, hi]^d ramenées sur [-1, 1]^d."""

    def __init__(self, in_dim: int, width: int, depth: int, lo: float = 0.0, hi: float = 1.0, out_dim: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
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


class MixedStateNet(nn.Module):
    """Forme mixte : u (bord fort) et flux sigma (N, d) ; forward rend u, comme StateNet."""

    def __init__(self, u: StateNet, flux: MLP):
        super().__init__()
        self.u = u
        self.flux_mlp = flux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u(x)

    def flux(self, x: torch.Tensor) -> torch.Tensor:
        return self.flux_mlp(x)


class RitzStateNet(nn.Module):
    """Forme de Ritz : état u et adjoint p (bords forts tous les deux) ; forward rend u, comme StateNet."""

    def __init__(self, u: StateNet, adjoint: StateNet):
        super().__init__()
        self.u = u
        self.adjoint = adjoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u(x)


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


def make_nets(cfg, problem):
    """Réseaux u (avec le flux en forme mixte, l'adjoint en forme de Ritz) et rho, sur cfg.run.device."""
    def mlp(out_dim: int = 1):
        return MLP(problem.dim, cfg.model.width, cfg.model.depth, problem.lo, problem.hi, out_dim)

    if cfg.loss.form not in ("strong", "mixed", "ritz"):
        raise ValueError(f"loss.form = {cfg.loss.form!r} : 'strong', 'mixed' ou 'ritz'")
    if cfg.loss.form == "ritz" and problem.dim != 1:
        raise ValueError("loss.form = 'ritz' : 1D seulement (matrice de Gram dense, certificat 1D)")
    u_net = StateNet(mlp(), problem.boundary_factor)
    if cfg.loss.form == "mixed":
        u_net = MixedStateNet(u_net, mlp(problem.dim))
    elif cfg.loss.form == "ritz":
        u_net = RitzStateNet(u_net, StateNet(mlp(), problem.boundary_factor))
    rho_net = DensityNet(mlp(), problem.v0)
    return u_net.to(cfg.run.device), rho_net.to(cfg.run.device)
