"""Figures d'un run de pinn2d.py, n'importe où : il suffit des fichiers live.npz (et capacity.npz) du run.

    python plot2d.py runs2d/latest                  # écrit live.png (et capacity.png) dans le dossier
    python plot2d.py runs2d/latest --watch          # fenêtre redessinée à chaque nouvel état, jusqu'à la fin du run
    python plot2d.py runs2d/latest --watch --png    # sans affichage graphique : réécrit live.png

Le run peut tourner ailleurs (container, machine distante) : il suffit que live.npz (et capacity.npz) arrivent
dans le dossier lu, par n'importe quelle copie ; --watch redessine à chaque nouvelle version.

Dépendances : numpy, matplotlib.
"""

import argparse
import os
import time
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

TERM_NAMES = ("J", "résidu", "Van der Waals", "volume")


def use_files():
    """Pas de fenêtre : figures écrites dans des fichiers (à appeler avant tout tracé)."""
    matplotlib.use("Agg")


def plt_close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def _save(fig, path: Path):
    tmp = path.with_name(path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=80)
    os.replace(tmp, path)   # atomique : un lecteur ne voit jamais une image à moitié écrite


def _field(fig, a, f, title, cmap="viridis", symmetric=False, vmin=None, vmax=None):
    """Champ f[i, j] en (x_i, y_j) sur [0,1]²."""
    if symmetric:
        vmax = float(np.max(np.abs(f))) or 1.0
        vmin = -vmax
    im = a.imshow(np.asarray(f).T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    a.set_title(title)
    a.set_aspect("equal")


def _circles(a, d):
    t = np.linspace(0.0, 2 * np.pi, 200)
    for (cx, cy), r in zip(np.asarray(d["centers"]), np.asarray(d["radii"]), strict=True):
        a.plot(cx + r * np.cos(t), cy + r * np.sin(t), "r--", lw=0.9)


def _isoline(a, rho, color="C0"):
    n = rho.shape[0]
    t = (np.arange(n) + 0.5) / n
    if rho.min() < 0.5 < rho.max():
        a.contour(t, t, np.asarray(rho).T, levels=[0.5], colors=color, linewidths=1.2)


def draw_live(d, path: Path | None = None, fig=None):
    """Figure de suivi à partir des données de live.npz (ou du même dictionnaire) ; rend la figure."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(22, 10))
    fig.clf()
    ax = fig.subplots(2, 4)
    rho, u, u_ex, u_star = (np.asarray(d[k], dtype=float) for k in ("rho", "u", "u_exact", "u_star"))
    a = ax[0, 0]
    _field(fig, a, rho, "ρ réseau (bleu : ρ = ½, rouge : cible)", cmap="gray_r", vmin=0.0, vmax=1.0)
    _circles(a, d)
    _isoline(a, rho)
    _field(fig, ax[0, 1], u, "u réseau")
    _field(fig, ax[0, 2], u - u_ex, f"u réseau - u exact (ρ du réseau)   erreur rel {float(d['u_error']):.2e}",
           cmap="RdBu_r", symmetric=True)
    _field(fig, ax[1, 2], u_ex - u_star, "u exact (ρ du réseau) - u*", cmap="RdBu_r", symmetric=True)
    _circles(ax[1, 2], d)
    t = np.asarray(d["terms"]).reshape(-1, len(TERM_NAMES))
    it = np.arange(1, len(t) + 1)
    a = ax[0, 3]
    if len(t):
        a.semilogy(it, np.maximum(t.sum(axis=1), 1e-16), "k", lw=1.5, label="totale")
        for k, name in enumerate(TERM_NAMES):
            if t[:, k].max() > 0:
                a.semilogy(it, np.maximum(t[:, k], 1e-16), lw=1.0, label=name)
    a.set_title("termes de la loss")
    c = np.asarray(d["checks"]).reshape(-1, 7)   # itération, J exact, erreur u, écart ρ, composantes, trous, Hausdorff
    a = ax[1, 0]
    a.semilogy(c[:, 0], c[:, 1], "C4o-", ms=3, label="J exact / J_ref (ρ du réseau)")
    a.semilogy(c[:, 0], c[:, 3], "C2s-", ms=3, label="écart RMS ρ - ρ*")
    if len(t):
        a.semilogy(it, np.maximum(t[:, 0], 1e-16), "C0", lw=0.8, label="J PINN / J_ref")
    a.set_title("objectif et écart au binaire")
    a = ax[1, 1]
    b0, b1 = np.asarray(d["target_betti"])
    a.plot(c[:, 0], c[:, 4], "C3o-", ms=3, label="composantes")
    a.plot(c[:, 0], c[:, 5], "C1^-", ms=3, label="trous")
    a.axhline(b0, color="C3", ls=":", lw=0.8)
    a.axhline(b1, color="C1", ls=":", lw=0.8)
    b = a.twinx()
    b.set_yscale("log")
    b.axhline(float(d["eps"]), color="C5", ls=":", lw=0.8)
    known = np.isfinite(c[:, 6])   # nan tant qu'il n'y a pas d'isoligne ρ = ½
    if known.any():
        b.plot(c[known, 0], c[known, 6], "C5s", ms=3, label="Hausdorff des interfaces")
        b.legend(loc="upper right", fontsize=8)
    a.set_title("topologie (pointillés : cible, ε)")
    for a in (ax[0, 3], ax[1, 0], ax[1, 1]):
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(loc="upper left" if a is ax[1, 1] else "best", fontsize=8)
        for s in np.atleast_1d(d["phase_starts"])[1:]:
            a.axvline(s, color="0.6", ls=":", lw=0.7)
        a.set_xlabel("itérations")
    ax[1, 3].axis("off")
    ax[1, 3].text(0.0, 1.0, str(d["summary"]), va="top", family="monospace", fontsize=9,
                  transform=ax[1, 3].transAxes)
    fig.suptitle(str(d["title"]))
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def draw_capacity(d, path: Path):
    """Figure du test de capacité à partir des données de capacity.npz (ou du même dictionnaire)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    rho = np.asarray(d["rho_fit"], dtype=float)
    _field(fig, ax[0], rho, f"ρ ajusté sur ρ_ε (erreur L2 rel {float(d['rho_error']):.1e})",
           cmap="gray_r", vmin=0.0, vmax=1.0)
    _circles(ax[0], d)
    _isoline(ax[0], rho)
    _field(fig, ax[1], d["u_eps"], "u exact pour ρ_ε")
    _field(fig, ax[2], np.asarray(d["u_fit"], dtype=float) - np.asarray(d["u_eps"], dtype=float),
           f"u réseau (forme forte, ρ_ε fixé) - u exact   erreur H1 rel {float(d['u_error_h1']):.1e}",
           cmap="RdBu_r", symmetric=True)
    fig.suptitle(str(d["title"]))
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _load(path: Path):
    """Contenu de path, ou None s'il manque ou est incomplet (copie en cours)."""
    try:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="dossier contenant live.npz (et capacity.npz)")
    p.add_argument("--watch", action="store_true", help="suivre le run jusqu'à sa fin")
    p.add_argument("--png", action="store_true", help="avec --watch : écrire live.png au lieu d'ouvrir une fenêtre")
    p.add_argument("--every", type=float, default=3.0, help="secondes entre deux lectures (--watch)")
    a = p.parse_args(argv)
    window = a.watch and not a.png
    if not window:
        use_files()
    import matplotlib.pyplot as plt

    live, cap = a.run_dir / "live.npz", a.run_dir / "capacity.npz"
    seen, cap_seen, fig = None, None, None
    while True:
        if cap.exists() and cap.stat().st_mtime != cap_seen:
            d = _load(cap)
            if d is not None:
                cap_seen = cap.stat().st_mtime
                draw_capacity(d, a.run_dir / "capacity.png")
                print(f"{a.run_dir / 'capacity.png'} : {d['title']}", flush=True)
        if live.exists() and live.stat().st_mtime != seen:
            d = _load(live)
            if d is not None:
                seen = live.stat().st_mtime
                fig = draw_live(d, None if window else a.run_dir / "live.png", fig)
                print(str(d["title"]), flush=True)
                if bool(d["done"]):
                    _save(fig, a.run_dir / "final.png")
                    print(f"{d['verdict']}   -> {a.run_dir / 'final.png'}", flush=True)
                    if window:
                        plt.show()
                    return
        if not a.watch:
            if seen is None and cap_seen is None:
                raise SystemExit(f"rien à tracer : ni {live} ni {cap}")
            return
        if window and fig is not None:
            plt.pause(a.every)
        else:
            time.sleep(a.every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
