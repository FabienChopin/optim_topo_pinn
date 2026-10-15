# Optimisation topologique 2D par champ de phase, PINN en résidu fort,
# minimisation jointe (approche "all-at-once" par pénalité).
#
# Problème
# --------
#     min_{u, rho}   J(u) = int_{[0.2, 0.3] x [0, 1]} u^2 dx
#     s.c.           -div(A(rho) grad u) = 1   sur Omega = [0, 1]^2,  u = 0 sur le bord
#                    int_Omega rho dx = V0  (pénalisé)
#     A(rho) = A1*rho + A2*(1 - rho),  rho in [0, 1]
#
# Loss totale (minimisée conjointement sur les deux réseaux)
#     L = J / J_ref
#       + beta   * int (-div(A grad u) - 1)^2 dx                      (résidu fort, L2^2)
#       + lambda * int (eps |grad rho|^2 + rho^2 (1-rho)^2 / eps) dx  (Van der Waals)
#       + mu     * (int rho dx - V0)^2                                (volume)
#
# J_ref = J du design homogène rho = V0 au P du cycle courant (résolution DF), recalculé
# à chaque cycle : J/J_ref < 1 <=> le design bat le matériau homogène.
#
# Discrétisation
#     - Gauss-Legendre tensoriel N_GAUSS x N_GAUSS sur [0, 1]^2 (résidu, VdW, volume)
#     - Gauss-Legendre tensoriel dédié sur la bande [J_A, J_B] x [0, 1] (objectif J)
#     - Strong BC : u(x, y) = x (1 - x) y (1 - y) N_u(x, y)
#     - rho(x, y) = sigmoid(N_rho(x, y))
#
# Fichiers
#     parametres.py : tous les paramètres
#     pinn.py       : quadrature, J_ref, réseaux, losses, solveur DF de référence
#     figures.py    : figures de résultats et comparaison PINN / DF
#     ce fichier    : journal et boucle d'entraînement (à lancer)
import sys
import time
import numpy as np
import torch
from parametres import (SEED, DEVICE, P_MAX, P_STEP, V0, EPS0, LAMBDA, MU, BETA, N_GAUSS, WIDTH, DEPTH,
                        ADAM_ITERS, ADAM_LR, LOG_EVERY, N_CYCLES, EPS_FACTOR, LBFGS_CHUNK,
                        MAX_CHUNKS, TOL_RHO, TOL_RES, TOL_RHO_FINAL, TOL_RES_FINAL,
                        TOL_STAGNATION, OUT_MODELS, LOG_FILE)
from pinn import X, W, XJ, j_homogene, UNet, RhoNet, compute_losses
from figures import plot_results, compare_fd

_ = torch.manual_seed(SEED)
# ----------------------------------------------------------------------------
# Journal : tout ce qui s'affiche (print, erreurs) est aussi écrit dans LOG_FILE
# ----------------------------------------------------------------------------
class Tee:
    """Duplique un flux de sortie dans un fichier, vidé à chaque écriture."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.tee_file = log_file
    def write(self, data):
        self.stream.write(data)
        self.tee_file.write(data)
        self.tee_file.flush()
    def flush(self):
        self.stream.flush()
        self.tee_file.flush()

while hasattr(sys.stdout, "tee_file"):   # script recollé : on revient aux flux d'origine
    sys.stdout = sys.stdout.stream

while hasattr(sys.stderr, "tee_file"):
    sys.stderr = sys.stderr.stream

_log_handle = open(LOG_FILE, "a", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log_handle)
sys.stderr = Tee(sys.stderr, _log_handle)
print(f"\n########## Lancement {time.strftime('%Y-%m-%d %H:%M:%S')}   SEED = {SEED} ##########")
print(f"P = 1 -> {P_MAX} (pas {P_STEP})  V0 = {V0}  EPS0 = {EPS0}  LAMBDA = {LAMBDA}  MU = {MU}  BETA = {BETA}  "
      f"N_GAUSS = {N_GAUSS}  WIDTH x DEPTH = {WIDTH} x {DEPTH}  N_CYCLES = {N_CYCLES}")
# ----------------------------------------------------------------------------
# Réseaux et historique
# ----------------------------------------------------------------------------
u_model = UNet().to(DEVICE)
rho_model = RhoNet().to(DEVICE)
params = list(u_model.parameters()) + list(rho_model.parameters())

history = {k: [] for k in ["total", "J/J_ref", "residu", "vdw", "vol_pen", "volume", "gris"]}
adam_ends = []   # indices de fin d'Adam (un par cycle), pour les figures
def record(parts):
    for k in history:
        history[k].append(parts[k])

def log(tag, it, parts):
    print(f"[{tag} {it:6d}] total={parts['total']:.4e}  J/Jref={parts['J/J_ref']:.4e}  "
          f"res={parts['residu']:.3e}  vdw={parts['vdw']:.3e}  "
          f"vol={parts['volume']:.4f}  gris={parts['gris']:.3f}")

# ----------------------------------------------------------------------------
# Entraînement : continuation sur EPS, chaque palier arrêté sur stabilité
# ----------------------------------------------------------------------------
print(f"Device : {DEVICE}   points de quadrature : {X.shape[0]} (+ {XJ.shape[0]} pour J)")
EPS = EPS0
for i in range(N_CYCLES):
    last_cycle = (i == N_CYCLES - 1)
    tol_rho = TOL_RHO_FINAL if last_cycle else TOL_RHO
    tol_res = TOL_RES_FINAL if last_cycle else TOL_RES
    P = min(P_MAX, 1.0 + P_STEP * i)                # continuation SIMP
    print(f"\n===== Cycle {i}   {time.strftime('%H:%M:%S')}   EPS = {EPS:.3e}   P = {P:g}   (tol aire modifiée {tol_rho:.0e}, tol ||r|| {tol_res:.0e}) =====")
    J_REF = j_homogene(P)                             # design homogène rho = V0 au P courant
    print(f"J_ref (rho homogène = V0, P = {P:g}) = {J_REF:.6e}")
    t0 = time.time()
    n_adam = ADAM_ITERS if i == 0 else 0          # Adam au premier palier seulement
    opt_adam = torch.optim.Adam(params, lr=ADAM_LR)
    for it in range(1, n_adam + 1):
        opt_adam.zero_grad()
        total, parts = compute_losses(u_model, rho_model, EPS, P, J_REF)
        total.backward()
        opt_adam.step()
        record(parts)
        if it % LOG_EVERY == 0 or it == 1:
            log("Adam", it, parts)
    adam_ends.append(len(history["total"]))
    # Un seul L-BFGS par palier (historique conservé d'un tronçon à l'autre)
    opt_lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_CHUNK,
                                  max_eval=int(1.25 * LBFGS_CHUNK),
                                  history_size=50, tolerance_grad=1e-11,
                                  tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
    n_eval = [0]
    def closure():
        opt_lbfgs.zero_grad()
        total, parts = compute_losses(u_model, rho_model, EPS, P, J_REF)
        total.backward()
        record(parts)
        n_eval[0] += 1
        if n_eval[0] % LOG_EVERY == 0:
            log("LBFGS eval", n_eval[0], parts)
        return total
    with torch.no_grad():
        rho_old = rho_model(X)
    loss_old = compute_losses(u_model, rho_model, EPS, P, J_REF)[1]["total"]
    status = f"plafond de {MAX_CHUNKS} tronçons atteint"
    for k in range(1, MAX_CHUNKS + 1):
        _ = opt_lbfgs.step(closure)
        with torch.no_grad():
            rho_new = rho_model(X)
        d_rho = (W * (rho_new - rho_old).abs()).sum().item()   # aire du design qui a changé
        rho_old = rho_new
        _, cur = compute_losses(u_model, rho_model, EPS, P, J_REF)
        res_l2 = np.sqrt(cur["residu"])
        d_loss = abs(cur["total"] - loss_old) / abs(loss_old)
        loss_old = cur["total"]
        print(f"  [tronçon {k:3d}] aire modifiée = {d_rho:.2e}   ||r||_L2 = {res_l2:.2e}   "
              f"variation relative loss = {d_loss:.2e}")
        if d_rho < tol_rho and res_l2 < tol_res:
            status = "stable (design et résidu sous tolérance)"
            break
        if d_loss < TOL_STAGNATION:
            status = "stagnation de la loss"
            if res_l2 >= tol_res:
                status += " avec ||r|| au-dessus de la tolérance : augmenter BETA"
            break
    print(f"Arrêt du palier : {status}")
    print(f"Temps du cycle : {time.time() - t0:.1f} s")
    _, final = compute_losses(u_model, rho_model, EPS, P, J_REF)
    log("Final", 0, final)
    print(f"J physique = {final['J/J_ref'] * J_REF:.6e}   "
          f"||r||_L2 = {np.sqrt(final['residu']):.3e}   "
          f"volume = {final['volume']:.5f} (cible {V0})")
    out_models = f"{i}_{OUT_MODELS}"
    torch.save({"u": u_model.state_dict(), "rho": rho_model.state_dict(), "eps": EPS, "p": P},
               out_models)
    fig_name = plot_results(i, P, u_model, rho_model, history, adam_ends)
    print(f"Figures : {fig_name}   Modèles : {out_models}")
    fig_fd, J_nn, J_fd, J_seuil, err_u = compare_fd(i, EPS, P, J_REF, u_model, rho_model)
    print(f"Comparaison DF : {fig_fd}   J_PINN/J_ref = {J_nn / J_REF:.4e}   "
          f"J_DF/J_ref = {J_fd / J_REF:.4e}   J_DF seuillé/J_ref = {J_seuil / J_REF:.4e}   erreur L2 relative sur u = {err_u:.3e}")
    EPS *= EPS_FACTOR
