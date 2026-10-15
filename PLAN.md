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
| 0.10 (grille 50) | 0.2803 | 0.2275 (0.496) |
| 0.15 (grille 50) | 0.3741 | 0.2870 (0.496) |

Avec un filtre de la taille de l'interface du PINN au moment où P monte (~0.15), SIMP donne
la même forme que le PINN R0 : douve épaisse et bombée à droite de la bande. L'échelle de
longueur au moment de la montée en P fixe la topologie.

## Résultats
- R0 (QUICK, SEED = 3) : douve épaisse ("boule") à droite de la bande, J_DF/J_ref = 0.93,
  seuillé 0.57 (volume 0.38), gris 0.65. Bonne idée, échelle grossière.

- P_AFTER_EPS (QUICK, SEED = 3) : dégradé lisse sans paroi ; J/J_ref 0.53 à P = 1 (SIMP 0.31)
  puis 1.17 à P = 3. Le PINN optimise mal dès le problème relaxé.

## 1D (`pinn_1d.py`, évaluateur exact, quelques dizaines de secondes par run)
- tracking : SIMP 0.003, PINN reste une rampe sigmoïde unique (0.77).
- two_bands (V0 = 0.5) : peu instructif (SIMP 0.90 à P = 3, design optimal gris). Avec
  sigmoid(N/eps), effondrement à EPS = 0.01 (rho = 0 partout, sans retour) ; pas avec sigmoid(N).
- band, effet de BETA (J exact / J_ref) :

  | BETA | P = 1 | P = 3 | couche isolante après la bande |
  |---|---|---|---|
  | SIMP | 0.19 | 0.22 | oui |
  | 1e4 | 0.48 | 0.60 | non (creux lisse vers 0.2) |
  | 1e3 | 0.40 | 0.60 | partielle, instable |
  | 1e2 | 0.25 | 0.27 | oui, forme proche de SIMP |

  Avec BETA = 1e4, la pénalité sur le résidu rend les interfaces nettes trop chères (u
  difficile à représenter) : la loss préfère un design lisse. J est exact : le gain avec
  BETA = 1e2 n'est pas un artefact du résidu.

## Prochain run 2D
`QUICK = True`, SEED = 3, `BETA = 1e2` (le reste par défaut). Vérifier ||r|| et J_DF.

## Ancien : ordre des continuations inversé
`QUICK = True`, `P_AFTER_EPS = True`, SEED = 3 : EPS 0.1 -> 0.016 à P = 1 (5 paliers), puis
P 1.5 -> 3 à EPS = 0.016 (9 cycles, ~18 min). Question : la paroi fine apparaît-elle ?

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
