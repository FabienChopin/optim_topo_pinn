# phasefield : optimisation topologique par champ de phase et PINN

Problème, sur Ω = [0, 1]^d avec d = 1 ou 2 :

```
min_{u, ρ}  J = ∫_bande u² dx,  bande = [0.2, 0.3] × [0, 1]^(d-1)
s.c.        -div(A(ρ) ∇u) = 1 sur Ω,  u = 0 sur ∂Ω
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
- Continuation en phases (`continuation.q_values` non vide), pour faire apparaître les détails puis finir en noir et blanc :
  1. **topologie** : `n_stages` paliers à q = `q0`, ε de `eps0` à `eps_topo` (≈ 2 à 3 `eps_min`). Le gris est permis et montre où l'optimum veut de la matière ;
  2. **binarisation** : un palier par valeur de `q_values` (géométrique, l'effet de RAMP est logarithmique en q), à ε = `eps_topo`. Les zones grises se cassent en barres ;
  3. **netteté** : `n_sharp_stages` paliers à q = `q_values[-1]`, ε de `eps_topo` à `eps_min` (≈ 1.5 × l'écart entre points).

  `λ` fixe la taille des détails (Van der Waals ≈ λ/3 × périmètre) ; ε ne sert qu'à les rendre nets.
  ```bash
  python run.py --set continuation.eps_topo=1e-2 --set "continuation.q_values=[1, 3, 10, 30, 99, 300]" \
                --set continuation.n_sharp_stages=3 --set continuation.n_stages=6
  ```

En 1D, la solution exacte est connue pour tout ρ (A u' = C - x). Elle sert à trois choses :
- calculer J_ref ;
- vérifier u et le vrai J du design courant pendant l'entraînement ;
- calculer deux optimums de référence. L'**optimum binaire** est la cible : ρ ∈ {0, 1}, avec la position des interfaces optimisée. L'**optimum relaxé** autorise le gris ; il est calculé pour la loi finale (`q_max`, ou `q_values[-1]` en mode en phases). S'il est meilleur que le binaire, c'est que le gris est « moins cher » que tout design 0/1 avec cette loi A(ρ).

## Test 1D de microstructure : laminage d'un matériau gris

```bash
python run.py --config configs/laminage.toml
python run.py diagnose runs/latest
```

Avec l'objectif de bande, l'optimum 1D n'a que 2 interfaces : il n'y a pas de détail à découvrir. Ce test en a un :

```
J = ∫ (u - u*)² dx      (problem.objective = "target")
```

u* est la solution exacte pour une densité grise ρ* = V0 + amp sin(2π k x) en loi harmonique (`target_amp`, `target_periods`). En 1D, la loi harmonique (q = A1/A2 - 1) est le mélange exact en série. Cela donne trois références exactes :
- **optimum relaxé** : ρ = ρ*, J = 0 ;
- **binaire** : un laminé de N couches qui suit ρ* donne J(N) ≈ 0.227 / N² / J_ref. Plus il y a de couches, mieux c'est ;
- **avec le périmètre** : chaque interface coûte λ/3 (Van der Waals d'un profil optimal). Le laminé optimal minimise J(N)/J_ref + λ/3 × interfaces, d'où N* = 5, 9, 19, 41 pour λ = 1e-2 … 1e-5.

Le test met tout à l'épreuve à la fois. λ fixe le nombre de couches visé. ε doit être assez petit pour la couche la plus fine, d'épaisseur ≈ (V0 - amp) / N*. Il faut assez de points par interface, et un réseau ρ capable de représenter 2N* interfaces raides. Pour la loi : à q = 0, un gris lisse donne J = 0 sans interface ; à q = 99, gris et laminé sont équivalents ; au-delà, le gris est pénalisé. Enfin, l'optimiseur doit faire naître les couches.

## 2D

```bash
python run.py --config configs/2d.toml
```

La physique, le réseau et la loss sont les mêmes qu'en 1D. Ce qui change :
- **Référence** : il n'y a plus de solution exacte. u et le vrai J du design courant sont vérifiés par un solveur volumes finis sur `output.n_grid`² cellules (`fv2d.py`). Les faces utilisent la moyenne harmonique de A et le bord est à une demi-maille du centre. Convergence à l'ordre 2 vérifiée sur le cas homogène, gradient adjoint vérifié par différences finies. J_ref vient du même solveur.
- **Optimums de référence** : on ne peut plus les calculer exactement. Ils sont remplacés par une optimisation topologique classique par densité (filtre, RAMP avec continuation en q, gradient adjoint, section `[reference]`). La « référence binaire » est le meilleur design seuillé au volume V0. Elle est de bonne qualité, mais rien ne garantit que ce soit l'optimum.
- **Figure** : elle montre les cartes de ρ (contour de la référence binaire en orange), de la référence, de u, et de l'écart entre u PINN et u volumes finis.
- **Gris** : le plancher de gris devient 4ε × longueur d'interface.
- **Coût** : il faut n² points de quadrature (300² = 90 000 par défaut), donc un GPU. `run.dtype = "float32"` est environ 5 fois plus rapide sur un GPU grand public (RTX). Garde `float64` sur un GPU de calcul (A100, H100).

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

## Diagnostic a posteriori : qu'est-ce qui a limité ?

```bash
python run.py diagnose runs/latest              # ou phasefield-diagnose ; --stage 8, --device cuda
```

Les réseaux de chaque palier sont rechargés, sans réentraîner. Le tableau (aussi écrit dans `diagnose.json`) donne par palier :

| Colonne | Mesure | Limitant si |
|---|---|---|
| `r` | netteté 1 / (4 ε \|∇ρ\|) sur ρ = 0.5 (1 = profil optimal sigmoïde(d / ε)) | (information) |
| `r_fit` | netteté d'une copie du réseau ρ recalée sur sigmoïde(d / ε) | > 1.5 : **réseau ρ** |
| `pts/if` | points de quadrature à travers une interface (4.4 ε / écart moyen) | < 4 : **points** |
| `rés f/e`, `ΔVdW`, `ΔJ` | résidu, Van der Waals et J sur une quadrature plus fine / d'entraînement | > 2, > 5 % : **points** |
| `excès` | gris - plancher 4 ε × interfaces | > 0.02 (et `r_fit` ≈ 1) : **loi (q)** si q < a1/a2 - 1, sinon **gris résiduel** (pénalisé mais pas résorbé) |
| `détail` | plus petit détail en largeurs d'interface (4.4 ε) | < 3 : **ε** |
| `plafond` | L-BFGS arrêté au nombre max d'itérations | oui : **optimiseur** |
| `erreur u` | erreur L2 sur u (exacte en 1D, volumes finis en 2D) | > 1e-2 : **réseau u** |

`r` grand avec `r_fit` ≈ 1 : le réseau saurait faire une interface nette, mais l'optimisation préfère une transition large, c'est-à-dire du gris.

## Organisation

```
configs/default.toml        paramètres (1D)
configs/2d.toml             paramètres (2D)
configs/laminage.toml       test 1D de microstructure (objectif cible, laminé)
src/phasefield/
  config.py                 lecture TOML + surcharges
  problem.py                problème 1D : A(ρ), source, bord, objectif, solution exacte
  exact1d.py                solution exacte 1D pour A constant par morceaux
  fv2d.py                   volumes finis 2D (solution de référence, gradient adjoint)
  references.py             références : optimums exacts et laminés (1D), optimisation par densité (2D)
  quadrature.py             Gauss composite sur [lo, hi]^d
  operators.py              gradient et divergence par autograd (dimension d)
  models.py                 MLP, u à bord fort, ρ = sigmoïde
  losses.py                 loss totale
  diagnostics.py            évaluation exacte du design courant
  diagnose.py               diagnostic a posteriori d'un run (facteur limitant par palier)
  plotting.py               figure de suivi
  train.py                  paliers (ε, q), Adam puis L-BFGS
  run.py                    point d'entrée, dossier du run
```

La physique (loi A, source, bord, objectif) est dans `BandProblem`. `BandProblem1D` et `BandProblem2D` y ajoutent leur solution de référence. Quadrature, opérateurs, réseaux et loss sont écrits en dimension d. Pour la 3D, il faut un solveur de référence 3D, puis les diagnostics et les panneaux de figure correspondants.

## Développement

```bash
ruff check src
```
