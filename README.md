# pinn1d : limites de l'optimisation topologique par PINN, en 1D

Un seul script, `pinn1d.py`, en JAX. On se donne un design binaire ρ* à N interfaces, et l'on cherche à le retrouver par optimisation d'un PINN (réseau u) couplé à un champ de phase (réseau ρ). Le script répond à deux questions :

1. **Le réseau peut-il représenter la cible ?** C'est le test de capacité, fait avant l'entraînement.
2. **L'optimiseur la retrouve-t-il ?** Il s'agit de retrouver le bon nombre d'interfaces, chacune à ε près.

## Installation

```bash
uv venv && uv pip install jax optax numpy matplotlib        # CPU
uv pip install "jax[cuda12]"                                # GPU NVIDIA (en plus)
```

JAX prend le GPU automatiquement s'il est disponible. `JAX_PLATFORMS=cpu` force le CPU.

## Démarrage

```bash
python pinn1d.py                                  # 1 interface, Adam 2000 puis L-BFGS 2000
display -update 2 runs1d/latest/live.png          # suivi en direct (ou tout visualiseur d'images)
```

Le log affiche le verdict du test de capacité, une ligne par mise à jour de la figure, puis un verdict final : `TOPOLOGIE RETROUVÉE` (bon nombre d'interfaces, chacune à moins de ε de sa cible) ou `TOPOLOGIE NON RETROUVÉE`.

Exemples courants :

```bash
python pinn1d.py --n-interfaces 3                            # 3 interfaces réparties sur [0.1, 0.9]
python pinn1d.py --interfaces 0.3 0.7                        # positions choisies
python pinn1d.py --opt adam:2000 lbfgs:2000 ng:200           # enchaîner les optimiseurs
python pinn1d.py --n-interfaces 6 --capacity-only            # seulement le test de capacité
python pinn1d.py --width 40 --depth 4 --eps 5e-3             # réseau plus gros, interfaces plus fines
python pinn1d.py --help                                      # toutes les options, avec leurs valeurs par défaut
```

## Le problème

Sur ]0, 1[, avec u(0) = u(1) = 0 :

```
-(A(ρ) u')' = f,      A(ρ) = a2 + (a1 - a2) ρ / (1 + q (1 - ρ))      (RAMP)
```

- **Cible.** ρ* est binaire, avec des interfaces aux positions données. u* est sa solution exacte, calculée sur une grille fine.
- **Identifiabilité.** En 1D, A u' = C − f x : u détermine A, donc ρ, point par point. ρ* est donc **l'unique optimum** (J = 0). Un échec vient de la méthode, jamais d'une ambiguïté du problème.
- **Valeur de q.** Par défaut, q = a1/a2 − 1 = 99 : c'est la loi harmonique, le mélange exact en série. Le gris n'y est pas favorisé.

## La méthode

Réseaux : `u = x (1 - x) N_u(x)` (bords exacts), `ρ = sigmoïde(N_ρ(x))`, N = MLP tanh (`--width`, `--depth`).

Loss en forme forte. C'est une somme de carrés, intégrée par quadrature de Gauss composite (`--elements` × `--gauss` points) :

```
L = ∫ (u - u*)² / J_ref                      J_ref : J du matériau uniforme ρ = V*
  + β ∫ (-(A(ρ) u')' - f)²                   résidu de l'EDP
  + λ ∫ (ε ρ'² + ρ² (1 - ρ)² / ε)            Van der Waals : interfaces de largeur ~ 4.4 ε
  + μ (∫ ρ - V*)²                            volume (μ = 0 par défaut, redondant ici)
```

Optimiseurs, enchaînés par `--opt nom:itérations` :

| nom | méthode |
|---|---|
| `adam` | Adam (optax), pas `--lr` |
| `lbfgs` | L-BFGS (optax), recherche linéaire de Wolfe |
| `ng` | gradient naturel de la loss de moindres carrés, c'est-à-dire Gauss-Newton amorti (Levenberg-Marquardt) sur u et ρ ensemble. Un pas coûte une Jacobienne complète : lent sur CPU, à réserver au GPU ou aux derniers pas. |

## Le test de capacité

Il est lancé avant chaque entraînement, sauf avec `--no-capacity`. Il se fait par Gauss-Newton, réseau par réseau :

- **ρ** : le réseau est ajusté sur le profil cible ρ_ε = sigmoïde(d/ε), qui est le profil optimal de Van der Waals.
- **u** : le réseau est entraîné en forme forte avec ρ = ρ_ε **fixé**, donné exactement et non par un réseau.

Le verdict est « OK » si l'erreur L² relative sur ρ est < 1e-2 avec le bon nombre d'interfaces, et si l'erreur H¹ relative sur u est < 1e-2. Sinon, la topologie visée est **hors de portée** de ces réseaux, quel que soit l'optimiseur : il faut un réseau plus gros, ou un ε plus grand. Les ajustements obtenus sont dans `capacity.png`.

## Sorties

Chaque run va dans `runs1d/<date>_<nom>_s<seed>/`, et `runs1d/latest` pointe vers le dernier run.

| fichier | contenu |
|---|---|
| `live.png` → `final.png` | ρ et u (réseau, cible, solution exacte du ρ courant), loss par terme, J exact, interfaces trouvées et erreur de position |
| `capacity.png` | ajustements du test de capacité |
| `log.txt` | journal |
| `metrics.json` | capacité, dernier état, verdict (`found`) |
| `config.json`, `params.npz` | options du run, paramètres finaux |

Une remarque sur les positions par défaut : avec `--n-interfaces N`, les interfaces sont en 0,1 + 0,8 (k + 0,6)/N et jamais en 0,5. À l'initialisation, les biais sont nuls, donc le MLP tanh est impair en 2x − 1 et ρ croise 0,5 en x = 0,5. Une cible centrée serait favorisée.
