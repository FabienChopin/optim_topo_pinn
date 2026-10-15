# Paramètres de l'optimisation topologique 2D par champ de phase (PINN, all-at-once).
# Modifier ici ; tous les autres fichiers importent ces valeurs.
import torch

SEED = 3                  # graine : lancer plusieurs graines et garder le meilleur J
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A1, A2 = 1.0, 0.01        # conductivités des deux phases
# A(rho) = A2 + (A1 - A2) * rho**P (SIMP). Continuation P = min(P_MAX, 1 + P_STEP * cycle) :
# P = 1 au départ (gradients sains, la topologie se forme), puis P_MAX pour chasser le gris.
P_MAX = 3.0
P_STEP = 0.5
V0 = 0.5                  # fraction volumique cible
EPS0 = 0.1                # épaisseur d'interface initiale
LAMBDA = 1e-3             # poids Van der Waals
MU = 100.0                # pénalisation du volume
BETA = 1e4                # poids du résidu fort (penalty all-at-once)
N_GAUSS = 300             # points de Gauss par direction sur [0, 1]^2 (-> 90 000 points)
N_GAUSS_JX = 40           # points de Gauss en x sur [J_A, J_B]
N_GAUSS_JY = 200          # points de Gauss en y sur [0, 1]
J_A, J_B = 0.2, 0.3       # bande de l'objectif : [J_A, J_B] x [0, 1]
WIDTH, DEPTH = 32, 3      # MLP : DEPTH couches cachées de WIDTH neurones
ADAM_ITERS = 2000         # Adam au premier palier seulement
ADAM_LR = 1e-3
LOG_EVERY = 500
N_CYCLES = 11             # paliers de continuation sur EPS
EPS_FACTOR = 10 ** (-0.1) # EPS <- EPS * EPS_FACTOR après chaque palier (0.1 -> 0.01)
# EPS final >= 2-3 x l'espacement des points de Gauss au centre (~ pi / (2 N_GAUSS)) :
# l'interface (largeur ~ 4.4 EPS) doit contenir quelques points. Plus fin -> augmenter N_GAUSS.
# Arrêt de chaque palier sur stabilité : L-BFGS par tronçons, contrôle entre tronçons
LBFGS_CHUNK = 1500        # itérations L-BFGS par tronçon
MAX_CHUNKS = 20           # plafond par palier (-> 30 000 itérations au plus)
MAX_CHUNKS_RAMP = 3       # plafond tant que P < P_MAX : design provisoire, inutile de l'affiner
TOL_RHO, TOL_RES = 1e-3, 1e-2              # paliers intermédiaires : aire modifiée, ||r||_L2
TOL_RHO_FINAL, TOL_RES_FINAL = 2e-4, 2e-3  # dernier palier
TOL_STAGNATION = 1e-6     # variation relative de la loss sur un tronçon en dessous de laquelle on s'arrête
N_PLOT = 201              # grille de post-traitement N_PLOT x N_PLOT
OUT_FIG = f"resultats_phase_field_pinn_2d_s{SEED}.png"
OUT_MODELS = f"modeles_phase_field_pinn_2d_s{SEED}.pt"
N_FD = 499                # noeuds intérieurs par direction pour la vérification DF
OUT_FIG_FD = f"comparaison_u_DF_s{SEED}.png"
LOG_FILE = f"log_phase_field_pinn_2d_s{SEED}.txt"   # copie de toute la sortie console
