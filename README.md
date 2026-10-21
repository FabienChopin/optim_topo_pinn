# phasefield : optimisation topologique par champ de phase et PINN, en JAX

C'est la réécriture en JAX de la branche `phase-field-1d` (PyTorch), dans le style des bancs `limites-1d` et `limites-2d` : un script de calcul, `phasefield.py`, et un script de figures, `plotphase.py`. On y ajoute le gradient naturel (`ng`).

## Le problème

Sur Ω = [0, 1]^d, avec d = 1 ou 2 :

```
min_{u, ρ}  J = ∫_bande u²,  bande = [0.2, 0.3] × [0, 1]^(d-1)
s.c.        -div(A(ρ) ∇u) = f sur Ω,  u = 0 sur ∂Ω,  ∫ρ = V0
            A(ρ) = a2 + (a1 - a2) ρ / (1 + q (1 - ρ))      (RAMP)
```

Réseaux : `u = Π x_i (1 - x_i) N_u(x)` (bords exacts), `ρ = sigmoïde(N_ρ(x))`, initialisé presque uniforme à V0. Les deux sont minimisés ensemble sur une seule loss de moindres carrés, intégrée par quadrature de Gauss composite tensorisée :

```
L = J / J_ref
  + β ∫ (-div(A(ρ) ∇u) - f)²                        résidu fort
  + λ ∫ (ε |∇ρ|² + ρ² (1 - ρ)² / ε)                 Van der Waals
  + μ (∫ρ - V0)²                                    volume
```

- J_ref est le J du matériau homogène ρ = V0 en loi linéaire (q = 0), pour que J/J_ref soit comparable d'un run à l'autre.
- **Paliers** : ε décroît géométriquement de `--eps0` à `--eps-min` en `--n-stages` paliers. q vaut `--q0` jusqu'au palier `--q-start`, puis monte linéairement jusqu'à `--q-max`. q = 0 donne la loi linéaire. q = a1/a2 − 1 = 99 donne la loi harmonique, le mélange exact en série en 1D. Au-delà, le gris est pénalisé. ε et q sont des arguments des fonctions compilées : changer de palier ne recompile rien.

## Optimiseurs

`--opt0` s'exécute au premier palier seulement (défaut : `adam:2000`), puis `--opt` à chaque palier (défaut : `lbfgs:1000`). Les deux options acceptent un enchaînement de `nom:itérations` :

| nom | méthode |
|---|---|
| `adam` | Adam (optax), pas `--lr` |
| `lbfgs` | L-BFGS (optax), recherche linéaire de Wolfe |
| `ng` | **gradient naturel** de la loss de moindres carrés, c'est-à-dire Gauss-Newton amorti (Levenberg-Marquardt) sur u et ρ ensemble : (JᵀJ + μ I) d = Jᵀr, avec un amortissement adapté au gain. JᵀJ (taille θ²) est accumulé par paquets de `--chunk` points, et la Jacobienne complète n'est jamais formée. Un pas coûte de l'ordre de (nombre de points) × θ² : à réserver aux derniers pas, et au GPU en 2D. |

Les quatre termes de la loss sont des sommes de carrés, volume compris (un résidu global). Le gradient naturel s'applique donc à la loss entière.

## Démarrage

```bash
uv venv && uv pip install jax optax numpy scipy matplotlib
python phasefield.py                               # 1D
python plotphase.py runs/latest --watch            # dans un autre terminal : figure en direct
```

Exemples :

```bash
python phasefield.py --opt lbfgs:1000 ng:20                  # gradient naturel en fin de chaque palier
python phasefield.py --opt0 adam:2000 ng:50 --opt ng:30      # gradient naturel seul après Adam
python phasefield.py --q-max 99 --q-start 4                  # loi linéaire 5 paliers, puis montée vers l'harmonique
python phasefield.py --dim 2                                 # 2D (GPU conseillé)
python phasefield.py --dim 2 --float32                       # ~5x plus rapide sur GPU grand public (RTX)
python phasefield.py --help                                  # toutes les options, avec leurs valeurs par défaut
```

Les valeurs par défaut dépendent de la dimension :

| option | 1D | 2D |
|---|---|---|
| `--elements` × `--gauss` | 100 × 10 | (100 × 3)² = 90 000 points |
| `--width` × `--depth` | 20 × 3 | 32 × 3 |
| `--eps-min`, `--n-stages` | 4e-3, 10 | 8e-3, 8 |

## Références et diagnostics

Une solution de référence, indépendante du PINN, évalue le design courant à chaque mise à jour :

- **1D** : solution exacte pour A constant par morceaux (A u' = C − f x) sur `--n-exact` cellules.
- **2D** : volumes finis à 5 points sur `--n-grid`² cellules, avec moyenne harmonique de A aux faces.

On en tire J exact (ρ du réseau), J du ρ seuillé à ½, le volume seuillé, l'erreur sur u, et le plancher de gris 4ε × (nombre d'interfaces en 1D, périmètre en 2D). Le gris G = ∫4ρ(1 − ρ) vaut 0 pour un design binaire.

Les optimums de référence sont calculés avant l'entraînement (`--no-references` pour s'en passer) :

- **1D** : l'**optimum binaire**, la cible, avec des positions d'interfaces optimisées par SLSQP depuis plusieurs départs. L'**optimum relaxé** autorise le gris, pour la loi finale q = `--q-max`. Il est calculé par gradient projeté, avec un gradient JAX à travers la solution exacte.
- **2D** : optimisation par densité classique, sur `--ref-cells`² cellules : filtre, RAMP avec continuation en q, gradient adjoint, L-BFGS-B avec lagrangien augmenté pour le volume. La « référence binaire » est le meilleur design seuillé au volume V0. Rien ne garantit que ce soit l'optimum.

## Sorties

Chaque run va dans `runs/<date>_<nom>_s<seed>/`, et `runs/latest` pointe vers le dernier run. Ctrl-C arrête proprement en sauvegardant l'état courant.

| fichier | contenu |
|---|---|
| `live.npz` | état courant : ρ, u PINN, u de référence, historique des termes, diagnostics, références. Marqué final à la fin du run. |
| `log.txt` | journal |
| `metrics.json` | références, bilan de chaque palier, état final |
| `config.json` | options du run |
| `params/stage_XX.npz`, `params/final.npz` | paramètres θ des réseaux, avec ε et q |
| `live.png`, `final.png` | figures, écrites par `plotphase.py`, ou par `phasefield.py` si matplotlib est présent |

`plotphase.py` ne demande que numpy et matplotlib. Pour un calcul sur un container GPU, il suffit de copier `live.npz` sur le PC et de lancer `python plotphase.py <dossier> --watch`.
