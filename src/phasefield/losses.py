"""Loss totale, minimisée conjointement sur u et rho :

    L = J / J_ref + beta int (-div(A(rho) grad u) - f)^2 + lam int (eps |grad rho|^2 + rho^2 (1 - rho)^2 / eps)
        + mu (int rho - v0)^2
"""


from .config import LossConfig
from .operators import div, grad
from .quadrature import Quadrature

TERMS = ("J", "residual", "vdw", "volume")   # termes pondérés, dans l'ordre de la loss


def compute_losses(problem, u_net, rho_net, quad: Quadrature, eps: float, w: LossConfig):
    """Loss totale (tenseur) et, en flottants : termes pondérés, volume et gris."""
    x = quad.points.clone().requires_grad_(True)
    u = u_net(x)
    rho = rho_net(x)
    flux = problem.conductivity(rho) * grad(u, x)                    # résidu fort, sur le flux
    residual = -div(flux, x) - problem.source(x)
    vdw = eps * (grad(rho, x) ** 2).sum(dim=1, keepdim=True) + rho ** 2 * (1.0 - rho) ** 2 / eps
    volume = quad.integrate(rho)
    terms = {
        "J": quad.integrate(problem.objective_density(x, u)) / problem.j_ref,
        "residual": w.beta * quad.integrate(residual ** 2),
        "vdw": w.lam * quad.integrate(vdw),
        "volume": w.mu * (volume - problem.v0) ** 2,
    }
    total = sum(terms.values())
    info = {k: v.item() for k, v in terms.items()}
    info["total"] = total.item()
    info["vol"] = volume.item()
    info["grey"] = quad.integrate(4.0 * rho * (1.0 - rho)).item()   # 0 : binaire, 1 : rho = 0.5 partout
    return total, info
