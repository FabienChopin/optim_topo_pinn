# pinn2d : limites de l'optimisation topologique par PINN, en 2D

Un seul script, `pinn2d.py`, en JAX. C'est la transposition en 2D du banc 1D (branche `limites-1d`). On se donne un design binaire ρ* fait de N disques, et l'on cherche à le retrouver par optimisation d'un PINN (réseau u) couplé à un champ de phase (réseau ρ). Le script répond à deux questions :

1. **Le réseau peut-il représenter la cible ?** C'est le test de capacité, fait avant l'entraînement.
2. **L'optimiseur la retrouve-t-il ?** Il s'agit de retrouver la bonne topologie (composantes, trous), avec des interfaces à moins de ε des cercles cibles.

## Installation

```bash
uv venv && uv pip install jax optax numpy scipy matplotlib  # CPU (matplotlib : pour les figures)
uv pip install "jax[cuda12]"                                # GPU NVIDIA (en plus)
```

JAX prend le GPU automatiquement s'il est disponible. `JAX_PLATFORMS=cpu` force le CPU. Les valeurs par défaut sont dimensionnées pour un GPU. Sur CPU, il faut réduire `--elements`, `--width` et `--fine`.

## Démarrage

```bash
python pinn2d.py                                  # 1 disque, Adam 2000 puis L-BFGS 2000
python plot2d.py runs2d/latest --watch            # dans un autre terminal : figure mise à jour en direct
```

Le log affiche le verdict du test de capacité, une ligne par mise à jour, puis un verdict final : `TOPOLOGIE RETROUVÉE` ou `TOPOLOGIE NON RETROUVÉE`.

Exemples courants :

```bash
python pinn2d.py --n-disks 4                                  # 4 disques sur une grille décalée
python pinn2d.py --disks 0.3 0.35 0.15  0.68 0.6 0.2          # disques choisis (cx cy r, en triplets)
python pinn2d.py --opt adam:2000 lbfgs:2000 ng:50             # enchaîner les optimiseurs
python pinn2d.py --n-disks 4 --capacity-only                  # seulement le test de capacité
python pinn2d.py --width 60 --depth 4 --eps 2e-2              # réseau plus gros, interfaces plus larges
python pinn2d.py --help                                       # toutes les options, avec leurs valeurs par défaut
```

## Le problème

Sur ]0, 1[², avec u = 0 au bord :

```
-div(A(ρ) ∇u) = f,      A(ρ) = a2 + (a1 - a2) ρ / (1 + q (1 - ρ))      (RAMP)
```

- **Cible.** ρ* vaut 1 (bon conducteur, a1) dans les disques et 0 (a2 = a1/100) dans la matrice. Les disques doivent être disjoints et intérieurs au carré. Sa topologie est notée (composantes, trous) = (N, 0).
- **Référence.** u* et la solution exacte du ρ courant sont calculées par éléments finis P1 sur une grille fine (`--fine` × `--fine` carrés coupés en deux triangles, A constant par carré), avec scipy.sparse.
- **Identifiabilité.** Contrairement au 1D, un seul chargement ne détermine plus A partout. ∇·(A ∇u*) = −f est une équation de transport pour A le long de ∇u*, qui dégénère là où ∇u* = 0. ρ* reste un optimum (J = 0), mais rien ne garantit qu'il soit le seul. Un échec peut donc venir du problème lui-même et pas seulement de la méthode. On le voit dans la figure : J exact petit avec ρ loin de ρ*.
- **Valeur de q.** Par défaut, q = a1/a2 − 1 = 99 : c'est la loi harmonique. Le gris n'y est pas favorisé.

## La méthode

Réseaux : `u = x (1-x) y (1-y) N_u(x, y)` (bords exacts), `ρ = sigmoïde(N_ρ(x, y))`, N = MLP tanh (`--width`, `--depth`).

Loss en forme forte. C'est une somme de carrés, intégrée par quadrature de Gauss composite tensorisée (`--elements`² éléments × `--gauss`² points) :

```
L = ∫ (u - u*)² / J_ref                      J_ref : J du matériau uniforme ρ = V*
  + β ∫ (-div(A(ρ) ∇u) - f)²                 résidu de l'EDP
  + λ ∫ (ε |∇ρ|² + ρ² (1 - ρ)² / ε)          Van der Waals : interfaces de largeur ~ 4.4 ε
  + μ (∫ ρ - V*)²                            volume (μ = 0 par défaut)
```

Optimiseurs, enchaînés par `--opt nom:itérations` :

| nom | méthode |
|---|---|
| `adam` | Adam (optax), pas `--lr` |
| `lbfgs` | L-BFGS (optax), recherche linéaire de Wolfe |
| `ng` | gradient naturel de la loss de moindres carrés, c'est-à-dire Gauss-Newton amorti (Levenberg-Marquardt) sur u et ρ ensemble. En 2D, la Jacobienne complète ne tient pas en mémoire. JᵀJ (taille θ², environ 10⁴ × 10⁴ pour les réseaux par défaut) est accumulé par paquets de `--chunk` points. À réserver au GPU et aux derniers pas. |

## Le test de capacité

Il est lancé avant chaque entraînement, sauf avec `--no-capacity`. Il se fait par Gauss-Newton, réseau par réseau :

- **ρ** : le réseau est ajusté sur ρ_ε = sigmoïde(d/ε), où d est la distance signée aux cercles.
- **u** : le réseau est entraîné en forme forte avec ρ = ρ_ε **fixé**, donné exactement et non par un réseau.

Le verdict est « OK » si l'erreur L² relative sur ρ est < 1e-2 avec la bonne topologie, et si l'erreur H¹ relative sur u (comparée à la solution EF pour ρ_ε) est < 1e-2. Sinon, la cible est **hors de portée** de ces réseaux.

## Diagnostics et verdict

À chaque mise à jour, sur la grille fine :

- **J exact** : J de la solution EF pour le ρ du réseau. C'est le vrai objectif, et non celui que voit le PINN.
- **Topologie** de {ρ > ½} : nombre de composantes (8-connexité) et de trous, c'est-à-dire les composantes du complémentaire, en 4-connexité, qui ne touchent pas le bord.
- **Hausdorff** : distance de Hausdorff entre l'isoligne ρ = ½ et les cercles cibles.

`TOPOLOGIE RETROUVÉE` demande la topologie de la cible **et** une distance de Hausdorff < ε.

## Figures et suivi : `plot2d.py`

`pinn2d.py` n'a pas besoin de matplotlib. À chaque mise à jour, il écrit l'état courant dans `live.npz` : ρ, u, u exact, u* sur la grille fine sous-échantillonnée d'un facteur 2, plus les loss et la topologie. `plot2d.py` en tire les figures, sur la machine qui calcule ou sur une autre. Il ne demande que numpy et matplotlib.

```bash
python plot2d.py runs2d/latest                    # une fois : live.png et capacity.png dans le dossier
python plot2d.py runs2d/latest --watch            # fenêtre redessinée à chaque nouvel état, jusqu'à la fin
python plot2d.py runs2d/latest --watch --png      # sans affichage graphique : réécrit live.png
```

**Calcul ailleurs (container GPU).** Il suffit de copier `live.npz` (et `capacity.npz`) du run vers le PC. `plot2d.py --watch` sur le dossier de destination redessine à chaque nouvelle version. Les écritures sont atomiques.

## Sorties

Chaque run va dans `runs2d/<date>_<nom>_s<seed>/`, et `runs2d/latest` pointe vers le dernier run.

| fichier | contenu |
|---|---|
| `live.npz` | état courant : champs ρ et u, loss par terme, J exact, topologie, Hausdorff. Marqué final à la fin du run. |
| `capacity.npz` | ajustements du test de capacité |
| `log.txt` | journal |
| `metrics.json` | capacité, dernier état, verdict (`found`) |
| `config.json`, `params.npz` | options du run, paramètres finaux |
| `live.png`, `capacity.png`, `final.png` | figures |

Une remarque sur les positions par défaut : avec `--n-disks N`, les disques sont sur une grille k × k décalée (k = ⌈√N⌉), et aucun n'est centré en (0,5 ; 0,5). À l'initialisation, les biais sont nuls, donc le MLP tanh est impair en (2x − 1, 2y − 1). Une cible centrée serait favorisée.
