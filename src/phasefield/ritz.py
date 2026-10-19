"""Deep Ritz avec gradient naturel en énergie (1D) : état u et adjoint p à rho fixé.

Les deux sont des minimiseurs d'énergie de la même forme a(v, w) = int A(rho) v' w' :
    E(v) = 1/2 a(v, v) - int g v,   g = f (état) ou dJ/du (adjoint).
Gradient : dE/dθ_i = a(v_θ - v*, ∂_i v_θ). Métrique naturelle : G_ij = a(∂_i v_θ, ∂_j v_θ). Le pas
d = G⁺ ∇E est la projection a-orthogonale de l'erreur v_θ - v* sur l'espace tangent : pas de Newton
en dimension infinie, indépendant du contraste de A. G est mal conditionnée (paramètres redondants) :
amortissement de Levenberg-Marquardt d = (G + μ λ_max I)⁻¹ ∇E. Gain r = baisse réelle de l'énergie /
baisse prédite par le modèle quadratique (∇E·d - d·G d / 2) : pas refusé et μ × 10 si r < 1/4 (la
linéarisation en θ ne vaut plus, même si l'énergie baisse), μ / 10 si r > 3/4.

Quadrature fixe : avec presque autant de paramètres que de points, le réseau peut osciller entre les
points et rendre l'énergie discrète aussi basse qu'il veut (-int g v n'est plus borné). Un pas n'est donc
accepté que si l'énergie baisse sur une quadrature de contrôle indépendante (plus fine), et le
certificat est calculé sur celle-ci.

Certificat (Prager-Synge, égalité en 1D) : avec sigma = c - int_0^x g, c fixé par int sigma / A = 0,
    ||v_θ - v*||_a^2 = int (sigma - A v_θ')^2 / A.
"""

import torch
from torch.func import functional_call, grad, jacrev, vmap

from .config import RitzConfig


def _cumint(quad, g: torch.Tensor) -> torch.Tensor:
    """int_0^x g aux points (1D) : exacte par élément pour une Quadrature ; sinon (points tirés au hasard)
    somme des poids précédents, demi-poids du point lui-même."""
    if hasattr(quad, "cumulative"):
        return quad.cumulative(g)
    order = torch.argsort(quad.points[:, 0])
    wg = (quad.weights * g)[order]
    out = torch.empty_like(wg)
    out[order] = torch.cumsum(wg, dim=0) - 0.5 * wg
    return out


def certificate(quad, a: torch.Tensor, g: torch.Tensor, dv: torch.Tensor) -> float:
    """||v_θ - v*||_a / ||v_θ||_a (le dénominateur borné inférieurement par 1e-30)."""
    big_g = _cumint(quad, g)
    c = quad.integrate(big_g / a) / quad.integrate(1.0 / a)
    err = quad.integrate((c - big_g - a * dv) ** 2 / a)
    norm = quad.integrate(a * dv ** 2)
    return float(torch.sqrt(err / norm.clamp_min(1e-30)))


class RitzSolver:
    """Gradient naturel en énergie sur les paramètres d'un StateNet (u = 0 au bord par construction)."""

    def __init__(self, net, cfg: RitzConfig):
        self.net, self.cfg = net, cfg
        self.names = [k for k, _ in net.named_parameters()]
        self.damping = cfg.damping

        def value(params, x):
            return functional_call(net, params, (x.reshape(1, -1),)).reshape(())

        self._du = grad(value, argnums=1)
        self._jac_u = vmap(jacrev(value, argnums=0), in_dims=(None, 0))
        self._jac_du = vmap(jacrev(self._du, argnums=0), in_dims=(None, 0))
        self._value = vmap(value, in_dims=(None, 0))
        self._dvalue = vmap(self._du, in_dims=(None, 0))

    def _params(self):
        return {k: p.detach() for k, p in self.net.named_parameters()}

    def _flat(self, jac: dict) -> torch.Tensor:
        n = next(iter(jac.values())).shape[0]
        return torch.cat([jac[k].reshape(n, -1) for k in self.names], dim=1)

    def fields(self, x):
        params = self._params()
        return self._value(params, x).reshape(-1, 1), self._dvalue(params, x).reshape(-1, 1)

    def energy(self, quad, a, g) -> torch.Tensor:
        v, dv = self.fields(quad.points)
        return quad.integrate(0.5 * a * dv ** 2 - g * v)

    def step(self, quad, a, g, check) -> bool:
        """Un pas de gradient naturel amorti ; False si aucun amortissement n'abaisse l'énergie."""
        x, w = quad.points, quad.weights
        params = self._params()
        v, dv = self.fields(x)
        jv = self._flat(self._jac_u(params, x))                  # (N, P)
        jdv = self._flat(self._jac_du(params, x)).reshape(len(x), -1)
        gradient = jdv.T @ (w * a * dv) - jv.T @ (w * g)         # (P, 1)
        gram = jdv.T @ (w * a * jdv)
        lam, vec = torch.linalg.eigh(gram)
        lam = lam.clamp_min(0.0)
        proj = vec.T @ gradient
        e0 = self.energy(*check)
        theta = torch.nn.utils.parameters_to_vector(self.net.parameters()).detach()
        for _ in range(self.cfg.max_tries):
            coef = proj / (lam[:, None] + self.damping * lam[-1])         # d dans la base propre
            predicted = (proj * coef).sum() - 0.5 * (lam[:, None] * coef ** 2).sum()
            torch.nn.utils.vector_to_parameters(theta - (vec @ coef).ravel(), self.net.parameters())
            gain = (e0 - self.energy(*check)) / predicted
            if gain > 0.75:
                self.damping = max(self.damping / 10.0, self.cfg.damping)
            if gain > 0.25:
                return True
            self.damping *= 10.0
        torch.nn.utils.vector_to_parameters(theta, self.net.parameters())
        self.damping = self.cfg.damping
        return False

    @torch.no_grad()
    def solve(self, quad, a: torch.Tensor, g: torch.Tensor, check: tuple) -> tuple[float, int]:
        """Pas sur quad (A et g aux points) jusqu'à certificat < tol (ou max_iters, ou plus de descente) ;
        check = (quadrature de contrôle, A, g) : acceptation des pas et certificat. Rend (certificat, pas)."""
        for it in range(self.cfg.max_iters + 1):
            cert = certificate(check[0], check[1], check[2], self.fields(check[0].points)[1])
            if cert < self.cfg.tol or it == self.cfg.max_iters or not self.step(quad, a, g, check):
                return cert, it
