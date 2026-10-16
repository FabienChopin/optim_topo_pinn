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

## Deep Ritz 1D, interface nette fixée en 0.5 (`ritz_1d_interface.py`, CPU, ~4 min)
-(A u')' = 1, A = 1 | 0.01, design fixé. Forme faible = min E(u) = int (A u'^2 / 2 - u),
un seul réseau (32 x 3), quadrature uniforme qui ignore l'interface. Erreurs sur grille fine.

| Cas | L2 rel u | énergie rel | flux loin de X_I |
|---|---|---|---|
| Gauss 49x6 fixe, Adam 3000 + L-BFGS | 3.4 (diverge) | 470 | 0.28 |
| Gauss 49x6 fixe, Adam 10000 | 2.8 (diverge) | 27 | 0.25 |
| Gauss 199x6 fixe, Adam 10000 | 3.5e-4 | 0.077 | 3e-3 |
| Gauss 199x6 fixe, Adam 3000 + L-BFGS | 2.9 (diverge) | 120 | 0.28 |
| Gauss 49x6 décalé au hasard, Adam 10000 | 7.5e-3 | 0.082 | 0.025 |
| Gauss 199x6 décalé au hasard, Adam 10000 | 5.0e-3 | 0.066 | 0.024 |
| Monte-Carlo 300, Adam 10000 | 2.1e-2 | 0.14 | 0.082 |

- Sur quadrature fixe, l'énergie discrète n'est pas bornée inférieurement : u monte par
  marches entre les points de Gauss côté isolant (pentes ×100 moins chères). E discrète
  -196 pour E vraie +1e4, écart de 173 sur u aux points mêmes. L-BFGS la trouve toujours.
- Points retirés à chaque itération (Gauss décalé) : stable, u à 0.5 % près.
- L'erreur d'énergie (~7 %) est à 99 % dans ±0.05 autour de l'interface : le réseau tanh
  lisse le coude de u (u' saute d'un facteur 100). Loin de l'interface, le flux est bon.
- Conséquence pour l'optimisation : pas de L-BFGS sur quadrature fixe avec Ritz.
- Pourquoi : l'énergie discrète ne voit u' qu'aux points ; toute fonction plate sur les points
  annule la partie quadratique, et -sum w u(x_k) n'est pas bornée (plateau de hauteur H :
  E_h = -H, E vraie ~ 900 H^2). Le Gauss décalé uniformément d'une fraction d'élément, modulo 1,
  est sans biais : E_s[E_{h,s}(u)] = E(u) pour tout u, donc Adam descend l'énergie vraie.

### Interface lissée (`python ritz_1d_interface.py epaisseur`)
rho = sigmoïde, A linéaire en rho, épaisseur w (rho 99 % -> 1 %).
- Solution exacte u_w -> u_0 (marche) : L2 et max en O(w) (2.9e-5 pour w = 1e-5), énergie en
  O(sqrt(w)) (4.4e-3 pour w = 1e-5).
- Ritz, erreur L2 par rapport à u_w : 1.6e-4 (w = 0.1), 2.2e-4 (1e-2), 2.1e-3 (1e-3), 5.0e-3
  (1e-5, identique à la marche à 3 chiffres près ; tous les cas aussi).
- Quadrature fixe + L-BFGS : correct à w = 0.1 (2.4e-5, interface résolue par les points),
  diverge dès w = 1e-2. Les falaises apparaissent là où le réseau peine à faire le coude.
- w = 1e-4, quadrature fixe dense (GPU, MLP 32 x 3) : L-BFGS ne diverge plus.

  | Quadrature | points dans l'interface | L2 / u_w | énergie rel | temps |
  |---|---|---|---|---|
  | Gauss 2001x6 fixe, Adam 3000 + L-BFGS | 0 | 2.6e-4 | 2.3e-2 | 93 s |
  | Gauss 10001x6 fixe, Adam 3000 + L-BFGS | 6 | 4.3e-4 | 1.6e-2 | 174 s |
  | Gauss 30001x6 fixe, Adam 3000 + L-BFGS | 18 | 4.9e-4 | 1.5e-2 | 407 s |
  | Gauss 10001x6 fixe, Adam 10000 | 6 | 2.9e-3 | 4.9e-2 | 184 s |
  | Gauss 199x6 décalé, Adam 10000 | 0 | 5.7e-3 | 6.8e-2 | 55 s |

  Ce qui compte : des points assez serrés pour que le réseau ne puisse plus dessiner de
  falaise entre eux (seuil entre 1 200 et 12 000 points pour 32 x 3, à revoir si le réseau
  grossit), pas la résolution de l'interface. Avec 2001x6, E_h < E* (-9e-6) : la triche
  commence mais reste négligeable. Test de contrôle : comparer E_h à l'énergie sur une
  quadrature décalée ; un écart net signale des falaises. Ne passe pas en 2D : 12 000 points
  par direction = 1.4e8 points (le 2D actuel : 90 000).

## 1D band : pourquoi SIMP reste gris (`band_1d_binaire.py`, numpy + scipy, ~1 min)
En 1D, u ne dépend que de 1/A, et un mélange fin des deux matériaux a pour conductivité la
moyenne harmonique (couches en série) : 1/A_mix = rho/A1 + (1 - rho)/A2. SIMP P = 3 donne au
gris A(0.5) = 0.134 contre 0.020 pour un vrai mélange : le gris est un matériau fictif,
meilleur que tout design 0/1, et l'optimiseur l'exploite. SIMP ne le pénalise près de
rho = 1 que si P >= A1/A2 = 100.

| Méthode | J/J_ref | gris | J seuillé (volume) |
|---|---|---|---|
| SIMP P = 3 (valeur 0.22 = fictive) | 0.2172 | 0.47 | 0.412 (0.397) |
| SIMP P = 100 | 0.3558 | 0.04 | 0.316 (0.515, volume violé) |
| loi mix, sans régularisation | 0.3260 | 0.000 | 0.3260 (0.500) |
| binaire, 2 interfaces optimisées (K = 1..4 testés) | 0.3260 | 0 | interfaces 0.299 / 0.799 |
| champ de phase sur grille, SIMP P = 3, EPS = 3h | 0.2420 | 0.32 | 0.387 (0.432) |
| champ de phase sur grille, mix, EPS = 3h | 0.3284 | 0.017 | 0.3266 (0.500) |

- L'optimum 0/1 est J/J_ref = 0.326 : conducteur sur [0, 0.3], isolant sur [0.3, 0.8],
  conducteur sur [0.8, 1]. Toute valeur plus basse (SIMP 0.19-0.22) est due au gris fictif.
- Avec la loi mix, le problème relaxé est déjà binaire ; le champ de phase y converge
  proprement (gris = largeur d'interface seulement). Avec SIMP, le double puits lutte contre
  le gain fictif du gris et déplace l'interface droite (0.865 au lieu de 0.80).
- PINN avec EPS -> 1e-8 : sigmoid(N / 1e-8) est un échelon, binaire par construction ;
  l'interface ne contient plus aucun point de Gauss, le design est figé. Juger sur J exact.
- `pinn_1d.py` : option `INTERP = "mix"` (J_ref inchangé, P_MAX forcé à 1), réseau 16 x 2.

### PINN 1D band, INTERP = "mix" (CPU, ~5 ms par évaluation)
| Run | J exact / J_ref | gris | seuillé (volume) |
|---|---|---|---|
| MAX_CHUNKS = 4 (plafond atteint à presque tous les cycles) | 0.450 | 0.38 | 0.404 (0.498) |
| MAX_CHUNKS = 20 : stable à 0.44 jusqu'à EPS = 0.02, puis effondrement à EPS = 0.016 | 5.37 | 0.06 | 4.29 (0.786) |

- Le PINN reste sur une rampe grise à droite (rho 0 -> 0.65 sur [0.6, 1]). Ce n'est pas un
  minimum du problème exact : L-BFGS-B sur grille, parti de cette rampe, va à 0.326 (binaire).
- À chaque changement d'EPS, BETA * résidu passe de ~0.005 à 1.5e4-3e4 : rho = sigmoid(N / eps)
  change alors que u reste figé. Avec la loi mix, A est très sensible à rho près de 1
  (dA/drho = 99 A^2 ; A(0.99) = 0.50) : la queue de la sigmoïde dans le conducteur suffit.
  Le L-BFGS passe le cycle à résorber ce pic ; à EPS = 0.016 il fait un pas qui détruit le design.
- Pistes : (1) à chaque nouvel EPS, réajuster u seul (rho figé) avant l'optimisation conjointe ;
  (2) formulation mixte u' = R(rho) q, q' = -1 avec R = 1/A linéaire en rho pour la loi mix
  (q régulier à travers les interfaces, problème bien conditionné).

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
