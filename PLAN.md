# Plan

## En cours : protocole de comparaison contrôlé, en mode rapide
`QUICK = True` dans `parametres.py` : arrêt à l'arrivée à P = 3 (5 cycles, EPS ~ 0.04),
résolution et budget par cycle inchangés, ~10-12 min par run. La question est qualitative :
le PINN trouve-t-il la paroi isolante de la référence SIMP ?
(Un premier essai avec 10 000 points et 10 fois moins d'évaluations n'a pas convergé :
J/J_ref ~ 2, gris ~ 0.85. Sur GPU, le coût d'une évaluation dépend peu du nombre de points.)

Un dossier par run, une console Python neuve par run, une seule modification par run.
Vérifier la graine dans l'en-tête du journal.

| Run | Réglages (en plus de QUICK = True) | Question |
|---|---|---|
| R0 | SEED = 3 | point de départ |
| R1 | SEED = 0, puis SEED = 69 | bruit dû à la graine |
| R2 | SEED = 3, RHO_SCALE_BY_EPS = False | effet de la paramétrisation de rho |
| R3 | SEED = 3, LAMBDA = 1e-3 | effet de lambda |

À relever (cycle 4, `4_shape_s<SEED>.npz` et figure) : paroi isolante présente ou non,
J_DF/J_ref et J_DF seuillé/J_ref (comparer à la référence SIMP r = 0.05 : 0.207 / 0.194).

Référence SIMP (`reference_simp_df.py`, évaluateur DF du PINN) :

| FILTER_RADIUS | J_DF/J_ref | J_DF seuillé/J_ref (volume) |
|---|---|---|
| 0.02 | 0.1786 | 0.1720 (0.498) |
| 0.03 | 0.1858 | 0.1807 (0.500) |
| 0.05 | 0.2068 | 0.1942 (0.500) |

Même topologie pour les trois rayons : massif conducteur à gauche englobant la bande, paroi
isolante fine juste à droite de la bande, doigt et bras conducteurs dans la moitié droite.
Le PINN n'a jamais produit cette paroi (meilleur J/J_ref ≈ 0.43).

## À venir : démarrer le PINN depuis la forme SIMP
But : savoir si la paroi isolante est une forme que la loss du PINN défavorise, ou seulement
une forme que son chemin d'optimisation ne trouve pas.

- Option `INIT_SHAPE = "shape_reference_simp.npz"` (None par défaut).
- Ajustement préalable du réseau rho sur cette forme (Adam sur l'écart des logits, quelques
  secondes), puis démarrage directement à P = P_MAX et EPS final (topologie donnée).
- Lecture :
  - la paroi tient et J_DF/J_ref ≈ 0.18-0.2 : forme stable pour le PINN, le problème est le
    chemin (exploration, premiers cycles) ;
  - la paroi se dissout et J remonte vers 0.4 : la loss du PINN défavorise cette forme
    (u difficile à représenter autour d'une paroi isolante fine, ou rho trop lisse). Piste :
    formulation faible ou mixte pour u, ou pondération locale du résidu.
