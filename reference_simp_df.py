# Référence SIMP classique, indépendante du PINN : volumes finis + adjoint + filtre de densité.
#
# Même problème que le PINN :
#     min J = int_{[J_A, J_B] x [0, 1]} u^2   s.c.  -div(A(rho) grad u) = 1, u = 0 au bord,
#                                                   int rho = V0
#     A = A2 + (A1 - A2) * rho_f**P,  rho_f = rho filtré (filtre conique de rayon FILTER_RADIUS)
# Même continuation P = 1 -> P_MAX que le PINN. Le design final est évalué avec le même
# évaluateur DF que le PINN (pinn.solve_fd, pinn.j_fd, J_ref = j_homogeneous(1.0)) :
# les J/J_ref sont directement comparables à ceux des .npz du PINN.
#
#     python reference_simp_df.py                  optimisation (quelques secondes)
#     python reference_simp_df.py --check-gradient vérification de l'adjoint par différences finies
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parametres import A1, A2, V0, J_A, J_B, P_MAX, P_STEP

N_SIMP = 100              # cellules par direction (h = 0.01 : la bande tombe sur des cellules)
FILTER_RADIUS = 0.03      # rayon du filtre de densité = échelle de longueur minimale du design
ITERS_PER_P = 50          # itérations par palier de P
ITERS_LAST_P = 150        # itérations au palier P = P_MAX
MOVE = 0.2                # déplacement maximal de rho par itération
OUT_SHAPE_REF = "shape_reference_simp.npz"
OUT_FIG_REF = "reference_simp.png"

# ----------------------------------------------------------------------------
# Volumes finis cellules centrées : K(A) u = 1, Dirichlet à une demi-cellule du bord
# ----------------------------------------------------------------------------
def assemble(A, h):
    """Matrice K (n*n, n*n) de -div(A grad .), moyenne harmonique de A aux faces."""
    n = A.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    diag = np.zeros((n, n))
    rows, cols, vals = [], [], []
    for sl_p, sl_q in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),   # faces en x
                       ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):  # faces en y
        ap, aq = A[sl_p], A[sl_q]
        a = 2.0 * ap * aq / (ap + aq) / h ** 2
        diag[sl_p] += a
        diag[sl_q] += a
        rows += [idx[sl_p].ravel(), idx[sl_q].ravel()]
        cols += [idx[sl_q].ravel(), idx[sl_p].ravel()]
        vals += [-a.ravel(), -a.ravel()]
    for wall in ((0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)):
        diag[wall] += 2.0 * A[wall] / h ** 2              # face de bord à distance h / 2
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n * n, n * n))

def objective_and_grad_A(A, h, band):
    """J = sum_band h^2 u^2 et dJ/dA par l'adjoint (K symétrique : une seule factorisation)."""
    n = A.shape[0]
    lu = spla.splu(assemble(A, h))
    u = lu.solve(np.ones(n * n)).reshape(n, n)
    J = float(h ** 2 * (band * u ** 2).sum())
    lam = lu.solve((2.0 * h ** 2 * band * u).ravel()).reshape(n, n)
    g = np.zeros((n, n))                                   # lam^T (dK/dA) u
    for sl_p, sl_q in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                       ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        ap, aq = A[sl_p], A[sl_q]
        prod = (u[sl_p] - u[sl_q]) * (lam[sl_p] - lam[sl_q]) / h ** 2
        g[sl_p] += 2.0 * aq ** 2 / (ap + aq) ** 2 * prod
        g[sl_q] += 2.0 * ap ** 2 / (ap + aq) ** 2 * prod
    for wall in ((0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)):
        g[wall] += 2.0 / h ** 2 * u[wall] * lam[wall]
    return J, -g, u

# ----------------------------------------------------------------------------
# Filtre de densité conique et paramétrisation SIMP
# ----------------------------------------------------------------------------
def build_filter(n, h, radius):
    """rho_f = (H rho) / Hs, poids max(0, radius - distance)."""
    idx = np.arange(n * n).reshape(n, n)
    reach = int(np.ceil(radius / h))
    rows, cols, vals = [], [], []
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            w = radius - h * np.hypot(di, dj)
            if w <= 0:
                continue
            src = idx[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)]
            dst = idx[max(0, di):n - max(0, -di), max(0, dj):n - max(0, -dj)]
            rows.append(src.ravel())
            cols.append(dst.ravel())
            vals.append(np.full(src.size, w))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n * n, n * n))
    return H, np.asarray(H.sum(axis=1)).ravel()

def objective_and_grad_rho(rho, p, h, band, H, Hs):
    """J et dJ/drho (rho : variables de design, avant filtre)."""
    n = band.shape[0]
    rho_f = (H @ rho.ravel() / Hs).reshape(n, n)
    A = A2 + (A1 - A2) * rho_f ** p
    J, dJ_dA, u = objective_and_grad_A(A, h, band)
    dJ_drho_f = dJ_dA * p * (A1 - A2) * rho_f ** (p - 1)
    return J, (H.T @ (dJ_drho_f.ravel() / Hs)).reshape(n, n), rho_f

def setup(n):
    h = 1.0 / n
    xc = (np.arange(n) + 0.5) * h
    band = np.broadcast_to(((xc > J_A) & (xc < J_B))[:, None], (n, n)).astype(float)
    H, Hs = build_filter(n, h, FILTER_RADIUS)
    return h, xc, band, H, Hs

# ----------------------------------------------------------------------------
# Vérification de l'adjoint
# ----------------------------------------------------------------------------
def check_gradient(n=20, p=3.0, n_checks=8, step=1e-6):
    h, _, band, H, Hs = setup(n)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.2, 0.8, (n, n))
    _, g, _ = objective_and_grad_rho(rho, p, h, band, H, Hs)
    worst = 0.0
    for _ in range(n_checks):
        i, j = rng.integers(0, n, 2)
        rp, rm = rho.copy(), rho.copy()
        rp[i, j] += step
        rm[i, j] -= step
        fd = (objective_and_grad_rho(rp, p, h, band, H, Hs)[0]
              - objective_and_grad_rho(rm, p, h, band, H, Hs)[0]) / (2 * step)
        err = abs(fd - g[i, j]) / max(abs(fd), 1e-30)
        worst = max(worst, err)
        print(f"  cellule ({i:2d}, {j:2d})  adjoint = {g[i, j]: .6e}  DF = {fd: .6e}  écart relatif = {err:.1e}")
    print(f"Écart relatif maximal : {worst:.1e}")

# ----------------------------------------------------------------------------
# Optimisation : gradient projeté normalisé, volume exact par bissection
# ----------------------------------------------------------------------------
def update(rho, g, H, Hs):
    """Pas de gradient normalisé, limité à MOVE, puis décalage eta tel que mean(rho_f) = V0."""
    n = rho.shape[0]
    target = rho - MOVE * g / np.abs(g).max()
    lo_b, hi_b = np.maximum(0.0, rho - MOVE), np.minimum(1.0, rho + MOVE)
    lo, hi = -1.0, 1.0
    for _ in range(60):
        eta = 0.5 * (lo + hi)
        new = np.clip(target + eta, lo_b, hi_b)
        if (H @ new.ravel() / Hs).mean() > V0:
            hi = eta
        else:
            lo = eta
    return new

def optimize():
    n = N_SIMP
    h, xc, band, H, Hs = setup(n)
    j_ref_fv = objective_and_grad_A(np.full((n, n), A2 + (A1 - A2) * V0), h, band)[0]   # P = 1
    rho = np.full((n, n), V0)
    p_values = []
    p = 1.0
    while p < P_MAX - 1e-12:
        p_values.append(p)
        p += P_STEP
    p_values.append(P_MAX)
    hist_J, hist_grey, p_ends = [], [], []
    t0 = time.time()
    for p in p_values:
        for _ in range(ITERS_LAST_P if p == P_MAX else ITERS_PER_P):
            J, g, rho_f = objective_and_grad_rho(rho, p, h, band, H, Hs)
            hist_J.append(J / j_ref_fv)
            hist_grey.append(float((4.0 * rho_f * (1.0 - rho_f)).mean()))
            rho = update(rho, g, H, Hs)
        p_ends.append(len(hist_J))
        print(f"P = {p:g}   J/J_ref (VF) = {hist_J[-1]:.4e}   gris = {hist_grey[-1]:.3f}   "
              f"volume = {rho_f.mean():.4f}   ({time.time() - t0:.1f} s)")
    rho_f = (H @ rho.ravel() / Hs).reshape(n, n)
    return xc, rho_f, hist_J, hist_grey, p_ends

def evaluate_like_pinn(xc, rho_f):
    """Évaluation avec l'évaluateur DF du PINN, sur sa grille G_FD."""
    from pinn import G_FD, solve_fd, j_fd, j_homogeneous   # import tardif : torch inutile pour --check-gradient
    g = G_FD
    gx, gy = np.meshgrid(g, g, indexing="ij")
    interp = RegularGridInterpolator((xc, xc), rho_f, bounds_error=False, fill_value=None)
    rho_grid = np.clip(interp(np.stack([gx.ravel(), gy.ravel()], axis=1)), 0.0, 1.0).reshape(len(g), len(g))
    binary = (rho_grid > 0.5).astype(float)
    wg = np.full(len(g), g[1] - g[0])
    wg[[0, -1]] *= 0.5
    j_ref = j_homogeneous(1.0)
    J_fd = j_fd(solve_fd(rho_grid, g, P_MAX), g)
    J_binary = j_fd(solve_fd(binary, g, P_MAX), g)
    vol = float((np.outer(wg, wg) * rho_grid).sum())
    vol_binary = float((np.outer(wg, wg) * binary).sum())
    np.savez(OUT_SHAPE_REF, x=g, y=g, rho=rho_grid, binary=binary.astype(np.uint8),
             volume=vol, volume_binary=vol_binary, J_DF=J_fd, J_DF_binary=J_binary, J_ref=j_ref,
             eps=np.nan, p=P_MAX, filter_radius=FILTER_RADIUS)
    print(f"Évaluateur DF du PINN : J_DF/J_ref = {J_fd / j_ref:.4e} (volume {vol:.4f})   "
          f"J_DF seuillé/J_ref = {J_binary / j_ref:.4e} (volume {vol_binary:.4f})   -> {OUT_SHAPE_REF}")
    return J_fd / j_ref, J_binary / j_ref

def draw(xc, rho_f, hist_J, hist_grey, p_ends, J_diffuse, J_binary):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    for a, img, title, kw in (
            (ax[0], rho_f, r"$\tilde\rho$ (filtré)", dict(cmap="gray_r", vmin=0.0, vmax=1.0)),
            (ax[1], np.log10(A2 + (A1 - A2) * rho_f ** P_MAX), rf"$\log_{{10}} A$   (P = {P_MAX:g})",
             dict(cmap="magma", vmin=np.log10(A2), vmax=np.log10(A1)))):
        im = a.imshow(img.T, origin="lower", extent=(0, 1, 0, 1), **kw)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        a.contour(xc, xc, rho_f.T, levels=[0.5], colors="tab:red", linewidths=1.0)
        for xb in (J_A, J_B):
            a.axvline(xb, color="tab:orange", ls="--", lw=1.2)
        a.set_title(title)
    it = np.arange(len(hist_J))
    ax[2].semilogy(it, hist_J, color="tab:blue", label="J/J_ref (VF)")
    ax[2].set_xlabel("itérations")
    ax[2].set_ylabel("J/J_ref", color="tab:blue")
    for n in p_ends[:-1]:
        ax[2].axvline(n, color="gray", ls="--", lw=1)
    axg = ax[2].twinx()
    axg.plot(it, hist_grey, color="tab:gray")
    axg.set_ylim(0.0, 1.0)
    axg.set_ylabel("gris", color="tab:gray")
    ax[2].grid(alpha=0.3)
    fig.suptitle(f"Référence SIMP   filtre r = {FILTER_RADIUS:g}   |   évaluateur DF du PINN : "
                 f"J_DF/J_ref = {J_diffuse:.4e}   J_DF seuillé/J_ref = {J_binary:.4e}")
    fig.tight_layout()
    fig.savefig(OUT_FIG_REF, dpi=100)
    plt.close(fig)
    print(f"Figure : {OUT_FIG_REF}")

if __name__ == "__main__":
    if "--check-gradient" in sys.argv:
        check_gradient()
    else:
        xc, rho_f, hist_J, hist_grey, p_ends = optimize()
        J_diffuse, J_binary = evaluate_like_pinn(xc, rho_f)
        draw(xc, rho_f, hist_J, hist_grey, p_ends, J_diffuse, J_binary)
