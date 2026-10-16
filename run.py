"""Lance un run sans installer le paquet (pas besoin de pip) :

    python run.py
    python run.py --set run.seed=4 --set loss.lam=1e-3
    python run.py --config configs/mon_essai.toml

Dépendances : Python >= 3.11, torch, numpy, scipy, matplotlib.
Les résultats vont dans runs/ sous le dossier courant (output.runs_dir).
"""

import sys
from pathlib import Path

if sys.version_info < (3, 11):  # noqa: UP036 (lancé sans pip, la contrainte du paquet ne joue pas)
    sys.exit(f"Python >= 3.11 requis (tomllib) ; version actuelle : {sys.version.split()[0]}")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from phasefield.run import main  # noqa: E402

if __name__ == "__main__":
    main()
