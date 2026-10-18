"""Lance un run sans installer le paquet (pas besoin de pip) :

    python run.py
    python run.py --set run.seed=4 --set loss.lam=1e-3
    python run.py --config configs/mon_essai.toml
    python run.py diagnose runs/latest          # diagnostic a posteriori (phasefield.diagnose)

Dépendances : Python >= 3.11, torch, numpy, scipy, matplotlib.
Les résultats vont dans runs/ sous le dossier courant (output.runs_dir).
"""

import sys
from pathlib import Path

if sys.version_info < (3, 11):  # noqa: UP036 (lancé sans pip, la contrainte du paquet ne joue pas)
    sys.exit(f"Python >= 3.11 requis (tomllib) ; version actuelle : {sys.version.split()[0]}")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    if sys.argv[1:2] == ["diagnose"]:   # python run.py diagnose runs/latest
        from phasefield.diagnose import main
        main(sys.argv[2:])
    else:
        from phasefield.run import main
        main()
