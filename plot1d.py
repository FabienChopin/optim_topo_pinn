"""Figures d'un run de pinn1d.py, n'importe où : il suffit des fichiers live.npz (et capacity.npz) du run.

    python plot1d.py runs1d/latest                  # écrit live.png (et capacity.png) dans le dossier
    python plot1d.py runs1d/latest --watch          # fenêtre redessinée à chaque nouvel état, jusqu'à la fin du run
    python plot1d.py runs1d/latest --watch --png    # sans affichage graphique : réécrit live.png

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


def draw_live(d, path: Path | None = None, fig=None):
    """Figure de suivi à partir des données de live.npz (ou du même dictionnaire) ; rend la figure."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(19, 9.5))
    fig.clf()
    ax = fig.subplots(2, 3)
    a = ax[0, 0]
    a.plot(d["x"], d["rho_star"], "k", lw=0.8, label="ρ* binaire")
    a.plot(d["x"], d["rho_eps"], "g--", lw=0.8, label="ρ_ε cible")
    a.plot(d["x"], d["rho"], "C0", label="ρ réseau")
    a.set_title("ρ(x)")
    a = ax[0, 1]
    a.plot(d["xu"], d["u_star"], "r", lw=0.9, label="u* cible")
    a.plot(d["xu"], d["u_exact"], "k", label="u exact (ρ du réseau)")
    a.plot(d["xu"], d["u"], "C1--", label="u réseau")
    a.set_title(f"u(x)   erreur relative {float(d['u_error']):.2e}")
    t = np.asarray(d["terms"]).reshape(-1, len(TERM_NAMES))
    it = np.arange(1, len(t) + 1)
    a = ax[0, 2]
    if len(t):
        a.semilogy(it, np.maximum(t.sum(axis=1), 1e-16), "k", lw=1.5, label="totale")
        for k, name in enumerate(TERM_NAMES):
            if t[:, k].max() > 0:
                a.semilogy(it, np.maximum(t[:, k], 1e-16), lw=1.0, label=name)
    a.set_title("termes de la loss")
    c = np.asarray(d["checks"]).reshape(-1, 6)   # itération, J exact, erreur u, écart ρ, interfaces, position
    a = ax[1, 0]
    a.semilogy(c[:, 0], c[:, 1], "C4o-", ms=3, label="J exact / J_ref (ρ du réseau)")
    a.semilogy(c[:, 0], c[:, 3], "C2s-", ms=3, label="écart RMS ρ - ρ*")
    if len(t):
        a.semilogy(it, np.maximum(t[:, 0], 1e-16), "C0", lw=0.8, label="J PINN / J_ref")
    a.set_title("objectif et écart au binaire")
    a = ax[1, 1]
    a.plot(c[:, 0], c[:, 4], "C3o-", ms=3, label="interfaces trouvées")
    a.axhline(int(d["n_target"]), color="C3", ls=":", lw=0.8)
    b = a.twinx()
    b.set_yscale("log")
    b.axhline(float(d["eps"]), color="C5", ls=":", lw=0.8)
    known = np.isfinite(c[:, 5])   # nan tant que le nombre d'interfaces n'est pas le bon
    if known.any():
        b.plot(c[known, 0], c[known, 5], "C5s", ms=3, label="erreur de position (max)")
        b.legend(loc="upper right", fontsize=8)
    a.set_title("topologie (pointillés : cible, ε)")
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]):
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(loc="upper left" if a is ax[1, 1] else "best", fontsize=8)
    for a in (ax[0, 2], ax[1, 0], ax[1, 1]):
        for s in np.atleast_1d(d["phase_starts"])[1:]:
            a.axvline(s, color="0.6", ls=":", lw=0.7)
        a.set_xlabel("itérations")
    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 1.0, str(d["summary"]), va="top", family="monospace", fontsize=9,
                  transform=ax[1, 2].transAxes)
    fig.suptitle(str(d["title"]))
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def draw_capacity(d, path: Path):
    """Figure du test de capacité à partir des données de capacity.npz (ou du même dictionnaire)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(d["x"], d["rho_star"], "k", lw=0.8, label="ρ* binaire")
    ax[0].plot(d["x"], d["rho_eps"], "g--", label="ρ_ε cible")
    ax[0].plot(d["x"], d["rho_fit"], "C0", label=f"réseau (erreur L2 rel {float(d['rho_error']):.1e})")
    ax[1].plot(d["xu"], d["u_eps"], "k", label="u exact pour ρ_ε")
    ax[1].plot(d["xu"], d["u_fit"], "C1--", label=f"réseau, forme forte (erreur H1 rel {float(d['u_error_h1']):.1e})")
    ax[0].set_title("ρ ajusté sur ρ_ε")
    ax[1].set_title("u en forme forte, ρ = ρ_ε fixé")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
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
