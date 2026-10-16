"""Opérateurs différentiels par autograd, en dimension quelconque (points indépendants)."""

import torch


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient de y (N, 1) par rapport à x (N, d) -> (N, d), graphe conservé."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def div(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Divergence d'un champ q (N, d) -> (N, 1)."""
    return sum(grad(q[:, k:k + 1], x)[:, k:k + 1] for k in range(x.shape[1]))
