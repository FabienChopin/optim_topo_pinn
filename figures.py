# Post-traitement : figures de résultats et comparaison PINN / différences finies.
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parametres import DEVICE, A1, A2, V0, LAMBDA, MU, BETA, J_A, J_B, N_PLOT, OUT_FIG, OUT_FIG_FD, OUT_LIVE, OUT_SHAPE
from pinn import XJ, WJ, G_FD, grad, divergence, conductivity, solve_fd, j_fd

def show_log_a(fig, a, rho_img, p):
    """log10 A(rho) : ce que voit l'EDP. Échelle [log10 A2, log10 A1]."""
    im = a.imshow(np.log10(conductivity(rho_img, p)), origin="lower", extent=(0, 1, 0, 1),
                  cmap="magma", vmin=np.log10(A2), vmax=np.log10(A1))
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    for xb in (J_A, J_B):
        a.axvline(xb, color="tab:cyan", ls="--", lw=1.2)
    a.set_title(rf"$\log_{{10}} A(\rho)$   (P = {p:g})")

# ----------------------------------------------------------------------------
# Aperçu en cours d'entraînement : rho et log10 A, fichier OUT_LIVE écrasé à chaque appel.
# À suivre depuis un autre terminal : display -update 5 live_rho_s<SEED>.png
# ----------------------------------------------------------------------------
def live_snapshot(rho_model, p, title):
    g = np.linspace(0.0, 1.0, N_PLOT)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), device=DEVICE)
    with torch.no_grad():
        rho_img = rho_model(pts).cpu().numpy().reshape(N_PLOT, N_PLOT).T
    fig, (ax, ax_a) = plt.subplots(1, 2, figsize=(11, 5.5))
    im = ax.imshow(rho_img, origin="lower", extent=(0, 1, 0, 1), cmap="gray_r", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for xb in (J_A, J_B):
        ax.axvline(xb, color="tab:orange", ls="--", lw=1.2)
    ax.set_title(r"$\rho$")
    show_log_a(fig, ax_a, rho_img, p)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    tmp = "tmp_" + OUT_LIVE                 # écriture puis renommage : jamais de fichier à moitié écrit
    fig.savefig(tmp, dpi=80)
    plt.close(fig)
    os.replace(tmp, OUT_LIVE)

def draw_figure(i, p, rho_img, u_img, r_img, history, adam_ends):
    fig, axd = plt.subplot_mosaic([["rho", "A", "u", "res"], ["loss", "loss", "loss", "vol"]],
                                  figsize=(24, 10))
    def show(a, img, title, extend="neither", **kw):
        im = a.imshow(img, origin="lower", extent=(0, 1, 0, 1), **kw)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, extend=extend)
        for xb in (J_A, J_B):
            a.axvline(xb, color="tab:orange", ls="--", lw=1.2)
        a.set_title(title)
        a.set_xlabel("x")
        a.set_ylabel("y")
    show(axd["rho"], rho_img, r"$\rho(x, y)$ (bande de J en pointillés)",
         cmap="gray_r", vmin=0.0, vmax=1.0)
    show_log_a(fig, axd["A"], rho_img, p)
    show(axd["u"], u_img, r"$u(x, y)$", cmap="viridis")
    # Résidu en log10|r|, échelle bornée aux percentiles 1-99 % (points aberrants saturés)
    log_r = np.log10(np.maximum(np.abs(r_img), 1e-16))
    r_lo, r_hi = np.percentile(log_r, [1, 99])
    show(axd["res"], log_r, r"$\log_{10}|-\nabla\cdot(A\nabla u) - 1|$ (percentiles 1-99 %)",
         extend="both", cmap="inferno", vmin=r_lo, vmax=r_hi)
    # Losses : loss totale et contributions pondérées, sur un seul axe
    it_axis = np.arange(len(history["total"]))
    ax_l = axd["loss"]
    total_hist = np.abs(np.asarray(history["total"]))
    ax_l.semilogy(it_axis, total_hist, color="k", lw=2.0, label="loss totale")
    sub = [("J/J_ref", 1.0, "J / J_ref", "tab:blue"),
           ("residual", BETA, f"β · résidu   (β = {BETA:g})", "tab:red"),
           ("vdw", LAMBDA, f"λ · Van der Waals   (λ = {LAMBDA:g})", "tab:green"),
           ("vol_pen", MU, f"μ · volume   (μ = {MU:g})", "tab:purple")]
    for key, weight, label, color in sub:
        y = weight * np.abs(np.asarray(history[key]))   # contribution pondérée à la loss totale
        ax_l.semilogy(it_axis, y, color=color, lw=1.0, alpha=0.8, label=label)
    # Bas de l'axe : PLOT_DYN_RANGE x (plus petite loss totale). En dessous, une
    # contribution est négligeable devant la totale : on la laisse sortir du cadre.
    PLOT_DYN_RANGE = 1e-6
    y_lo = PLOT_DYN_RANGE * total_hist[total_hist > 0].min()
    y_hi = 2.0 * max(total_hist.max(),
                     max(w * np.abs(np.asarray(history[k])).max() for k, w, _, _ in sub))
    ax_l.set_ylim(y_lo, y_hi)
    ax_l.set_ylabel("loss (contributions pondérées)")
    ax_l.set_xlabel("itérations (Adam | L-BFGS évaluations)")
    for n in adam_ends:
        ax_l.axvline(n, color="gray", ls="--", lw=1)
    ax_l.legend(fontsize=9, loc="best")
    ax_l.set_title("Losses")
    ax_l.grid(alpha=0.3)
    axv = axd["vol"]
    axv.plot(it_axis, history["volume"], color="tab:blue")
    axv.axhline(V0, color="k", ls="--", lw=1)
    for n in adam_ends:
        axv.axvline(n, color="gray", ls="--", lw=1)
    axv.set_ylabel(r"$\int \rho\,dx$", color="tab:blue")
    axg = axv.twinx()
    axg.plot(it_axis, history["grey"], color="tab:gray")
    axg.set_ylim(0.0, 1.0)
    axg.set_ylabel(r"gris $4\int \rho(1-\rho)\,dx$", color="tab:gray")
    axv.set_title(r"volume et gris")
    axv.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"{i}_{OUT_FIG}"
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname

def plot_results(i, p, u_model, rho_model, history, adam_ends):
    g = np.linspace(0.0, 1.0, N_PLOT)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    xp = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1),
                      device=DEVICE).requires_grad_(True)
    up = u_model(xp)
    rhop = rho_model(xp)
    rp = -divergence(conductivity(rhop, p) * grad(up, xp), xp) - 1.0
    def to_img(t):  # (N, 1) -> image (y, x) pour imshow(origin="lower")
        return t.detach().cpu().numpy().reshape(N_PLOT, N_PLOT).T
    return draw_figure(i, p, to_img(rhop), to_img(up), to_img(rp), history, adam_ends)

# ----------------------------------------------------------------------------
# Vérification : u par différences finies (rho du réseau figé) vs u du réseau
# ----------------------------------------------------------------------------
def draw_fd_figure(i, eps, p, j_ref, g, u_nn, u_fd, J_nn, J_fd, J_binary, vol_binary, rel_l2):
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    u_min, u_max = min(u_nn.min(), u_fd.min()), max(u_nn.max(), u_fd.max())
    def show(a, img, title, extend="neither", **kw):
        im = a.imshow(img.T, origin="lower", extent=(0, 1, 0, 1), **kw)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, extend=extend)
        for xb in (J_A, J_B):
            a.axvline(xb, color="tab:orange", ls="--", lw=1.2)
        a.set_title(title)
        a.set_xlabel("x")
        a.set_ylabel("y")
    show(ax[0], u_nn, r"$u$ PINN", cmap="viridis", vmin=u_min, vmax=u_max)
    show(ax[1], u_fd, r"$u$ différences finies", cmap="viridis", vmin=u_min, vmax=u_max)
    err = u_nn - u_fd
    e_max = np.percentile(np.abs(err), 99)   # échelle bornée au percentile 99 %
    show(ax[2], err, r"$u_{PINN} - u_{DF}$ (percentile 99 %)", extend="both",
         cmap="RdBu_r", vmin=-e_max, vmax=e_max)
    j_mid = len(g) // 2
    ax[3].plot(g, u_fd[:, j_mid], "k", lw=2, label="DF")
    ax[3].plot(g, u_nn[:, j_mid], "tab:red", ls="--", lw=1.5, label="PINN")
    ax[3].axvspan(J_A, J_B, color="tab:orange", alpha=0.2)
    ax[3].set_title(f"coupe y = {g[j_mid]:.3f}")
    ax[3].set_xlabel("x")
    ax[3].legend(loc="best")
    ax[3].grid(alpha=0.3)
    fig.suptitle(f"Cycle {i}   ε = {eps:.2e}   P = {p:g}   |   J_PINN/J_ref = {J_nn / j_ref:.4e}   "
                 f"J_DF/J_ref = {J_fd / j_ref:.4e}   (écart {abs(J_nn - J_fd) / J_fd:.2%})   "
                 f"J_DF seuillé/J_ref = {J_binary / j_ref:.4e} (volume {vol_binary:.4f})   |   "
                 f"erreur L2 relative sur u = {rel_l2:.3e}")
    fig.tight_layout()
    fname = f"{i}_{OUT_FIG_FD}"
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname

def compare_fd(i, eps, p, j_ref, u_model, rho_model):
    g = G_FD
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), device=DEVICE)
    with torch.no_grad():
        rho_grid = rho_model(pts).cpu().numpy().reshape(len(g), len(g))
        u_nn = u_model(pts).cpu().numpy().reshape(len(g), len(g))
        J_nn = float((WJ * u_model(XJ) ** 2).sum())
    u_fd = solve_fd(rho_grid, g, p)
    J_fd = j_fd(u_fd, g)
    # Design seuillé à 0.5 (rho dans {0, 1}) : prix de l'interface diffuse à l'EPS courant
    binary = (rho_grid > 0.5).astype(float)
    J_binary = j_fd(solve_fd(binary, g, p), g)
    wg = np.full(len(g), g[1] - g[0])               # poids des trapèzes
    wg[[0, -1]] *= 0.5
    vol_binary = float((np.outer(wg, wg) * binary).sum())
    # La forme elle-même (le livrable), relisible sans torch ni réseau : np.load(...)
    np.savez(f"{i}_{OUT_SHAPE}", x=g, y=g, rho=rho_grid, binary=binary.astype(np.uint8),
             volume=float((np.outer(wg, wg) * rho_grid).sum()), volume_binary=vol_binary,
             J_DF=J_fd, J_DF_binary=J_binary, J_ref=j_ref, eps=eps, p=p)
    rel_l2 = float(np.sqrt(((u_nn - u_fd) ** 2).mean() / (u_fd ** 2).mean()))
    fname = draw_fd_figure(i, eps, p, j_ref, g, u_nn, u_fd, J_nn, J_fd, J_binary, vol_binary, rel_l2)
    return fname, J_nn, J_fd, J_binary, vol_binary, rel_l2
