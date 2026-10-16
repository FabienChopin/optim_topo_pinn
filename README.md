# phasefield : optimisation topologique par champ de phase et PINN

Problème (1D pour l'instant, Ω = [0, 1]) :

```
min_{u, ρ}  J = ∫_{0.2}^{0.3} u² dx
s.c.        -(A(ρ) u')' = 1 sur Ω,  u(0) = u(1) = 0
            A(ρ) = A2 + (A1 - A2) ρ / (1 + q (1 - ρ)),  ρ ∈ [0, 1] (champ de phase, loi RAMP)
```

Les deux inconnues sont des réseaux : u est un PINN avec conditions au bord fortes,
u = x (1 - x) N_u(x), et ρ = sigmoïde(N_ρ(x)). Les deux sont minimisés ensemble sur une seule loss :

```
L = J / J_ref
  + β ∫ (-(A(ρ) u')' - 1)² dx                       résidu fort
  + λ ∫ (ε |ρ'|² + ρ² (1 - ρ)² / ε) dx              Van der Waals
  + μ (∫ ρ dx - V0)²                                volume
```

- J_ref est le J exact du matériau homogène ρ = V0.
- Les intégrales utilisent une quadrature de Gauss composite (1000 points par défaut). Les bornes de la bande sont alignées sur des bords d'éléments.
- ε décroît par paliers géométriques : Adam au premier palier, puis L-BFGS à chaque palier.
- Loi RAMP : q = 0 donne la loi linéaire (valeur par défaut). q = A1/A2 - 1 (= 99) donne la loi harmonique, c'est-à-dire des couches en série, le mélange exact en 1D. Au-delà, le gris est pénalisé. q peut monter par paliers : `continuation.q0`, `q_max`, `q_start`. Par exemple, `--set continuation.q_max=99 --set continuation.q_start=4` garde la loi linéaire pendant les 5 premiers paliers, puis monte q jusqu'à 99. J_ref reste calculé en loi linéaire (q = 0), pour que J/J_ref soit comparable d'un run à l'autre.

En 1D, la solution exacte est connue pour tout ρ (A u' = C - x). Elle sert à trois choses :
- calculer J_ref ;
- vérifier u et le vrai J du design courant pendant l'entraînement ;
- calculer deux optimums de référence. L'**optimum binaire** est la cible : ρ ∈ {0, 1}, avec la position des interfaces optimisée. L'**optimum relaxé** autorise le gris ; il est calculé pour la loi finale (q = `q_max`). S'il est meilleur que le binaire, c'est que le gris est « moins cher » que tout design 0/1 avec cette loi A(ρ).

## Installation

```bash
# avec uv
uv venv .venv && uv pip install -e ".[dev]"
# sans uv
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

**Sans installation** (supercalculateur, pas de pip) : il suffit d'avoir Python ≥ 3.11, torch, numpy, scipy et
matplotlib, puis de remplacer `phasefield-run` par `python run.py` (à la racine du dépôt) dans les commandes
ci-dessous. Le script ajoute `src/` au chemin de Python. Les runs vont dans `runs/` sous le dossier courant.

```bash
git clone -b phase-field-1d https://github.com/FabienChopin/optim_topo_pinn.git
cd optim_topo_pinn
python run.py --set run.seed=4
```

torch est pris sur PyPI. Pour une variante CPU seule ou une version précise de CUDA, installe torch d'abord :
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Lancer un run

```bash
phasefield-run                                         # configs/default.toml
phasefield-run --set run.seed=4 --set loss.lam=1e-3    # surcharges section.cle=valeur
phasefield-run --config configs/mon_essai.toml --set run.name=essai
```

Tous les paramètres et leur rôle sont dans [`configs/default.toml`](configs/default.toml).

Suivi pendant l'entraînement : la figure est réécrite toutes les `output.plot_every` évaluations, de façon atomique.

```bash
display -update 5 runs/latest/live.png
```

La figure contient six panneaux :
- ρ(x), avec les deux optimums de référence ;
- u PINN comparée à u exacte ;
- les termes de la loss multipliés par leurs poids ;
- J/J_ref : celui du PINN, le J exact du ρ courant, et le J exact du ρ seuillé à 0.5 ;
- le volume et le gris ;
- un résumé des paramètres et des indicateurs.

Gris : G = ∫ 4ρ(1 - ρ) dx, qui vaut 0 si le design est binaire et 1 si ρ = 0.5 partout. Une interface diffuse au profil optimal sigmoïde(x/ε) coûte 4ε. Le « plancher » 4ε × (nombre d'interfaces) est donc le gris d'un design binaire aux interfaces diffuses.

## Résultats

Chaque run crée un dossier `runs/<date>_<nom>_s<seed>/`, ignoré par git. `runs/latest` pointe toujours vers le dernier run. Le dossier contient :

| Fichier | Contenu |
|---|---|
| `config.toml` | configuration complète effective, à réutiliser avec `--config` |
| `log.txt` | journal |
| `live.png`, `final.png` | figure de suivi, puis figure finale |
| `metrics.json` | optimums de référence, bilan de chaque palier, état final |
| `models/` | réseaux à la fin de chaque palier et à la fin du run |

Ctrl-C arrête proprement : l'état courant est sauvegardé.

## Organisation

```
configs/default.toml        paramètres
src/phasefield/
  config.py                 lecture TOML + surcharges
  problem.py                problème 1D : A(ρ), source, bord, objectif, solution exacte
  exact1d.py                solution exacte pour A constant par morceaux
  references.py             optimum binaire et optimum relaxé
  quadrature.py             Gauss composite sur [lo, hi]^d
  operators.py              gradient et divergence par autograd (dimension d)
  models.py                 MLP, u à bord fort, ρ = sigmoïde
  losses.py                 loss totale
  diagnostics.py            évaluation exacte du design courant
  plotting.py               figure de suivi
  train.py                  paliers en ε, Adam puis L-BFGS
  run.py                    point d'entrée, dossier du run
```

Pour un autre objectif ou un autre problème (2D, 3D), il faut écrire une nouvelle classe à côté de `BandProblem1D`, avec les mêmes méthodes. Quadrature, opérateurs, réseaux et loss sont écrits en dimension d. Les diagnostics et les références exactes ne valent que pour le 1D.

## Développement

```bash
ruff check src
```
