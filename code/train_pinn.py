#!/usr/bin/env python3
"""train_pinn.py
Physics-Informed Neural Network (PINN) training for UV-laser polymer ablation.

Trains two model families and saves all predictions to fdm_output/pinn_results.npz:

1. Spatio-temporal PINN  (r̃, z̃, t̃) → T̂(°C)           [Fig 3, Fig 4]
   - Training data : FDM snapshots at pulses {1, 5, 10, 20}
   - Validation    : FDM snapshots at pulses {35, 50}
   - PDE loss      : cylindrical heat equation
   - BC  loss      : symmetry (r=0), Newton+SB cooling (z=0)

2. Ablation-diameter surrogate  Ep → D_abl               [Fig 5, Fig 6]
   - PINN : monotonicity + concavity constraints
   - ANN  : unconstrained MLP (same architecture)
   - Training Ep ≤ 150 µJ;  test / extrapolation Ep > 150 µJ

Runtime: ~6–12 min on CPU.
"""
from __future__ import annotations
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ─── reproducibility ────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('cpu')

ROOT    = Path(__file__).resolve().parent.parent
FDM_OUT = ROOT / 'fdm_output'

# ─── physical constants (must match export_training_data.py) ────
R_MAX    = 200e-6    # m domain radius
H_FILM   = 100e-6    # m film thickness
T_AMB    = 25.0      # °C
T_ABL    = 350.0     # °C
H_CONV   = 25.0      # W/(m²·K)
EMISS    = 0.95
SB       = 5.67e-8   # W/(m²·K⁴)
N_PULSES = 50
RHO_PP   = 910.0
RHO_PE   = 950.0
T_M_PP   = 165.0
T_M_PE   = 132.0


# ═══════════════════════════════════════════════════════════════════
# Differentiable material properties (PyTorch)
# ═══════════════════════════════════════════════════════════════════
def cp_pp_t(T: torch.Tensor) -> torch.Tensor:
    cb = 1780.0 + 570.0 / (1.0 + torch.exp(-(T - 140.0) / 50.0))
    return cb + 250.0 * torch.exp(-0.5 * ((T - 165.0) / 8.5) ** 2)

def k_pp_t(T: torch.Tensor) -> torch.Tensor:
    kc = (0.22 * 298.15 / torch.clamp(T + 273.15, min=200.0)).clamp(max=0.24)
    km = (0.155 - 3e-5 * (T - T_M_PP).clamp(min=0.0)).clamp(min=0.13)
    fm = torch.sigmoid((T - T_M_PP) / 10.0)
    return kc * (1.0 - fm) + km * fm

def cp_pe_t(T: torch.Tensor) -> torch.Tensor:
    cb = 1900.0 + 400.0 / (1.0 + torch.exp(-(T - 110.0) / 40.0))
    return cb + 370.0 * torch.exp(-0.5 * ((T - 132.0) / 7.0) ** 2)

def k_pe_t(T: torch.Tensor) -> torch.Tensor:
    kc = (0.45 * 298.15 / torch.clamp(T + 273.15, min=200.0)).clamp(max=0.50)
    km = (0.35 - 2e-5 * (T - T_M_PE).clamp(min=0.0)).clamp(min=0.30)
    fm = torch.sigmoid((T - T_M_PE) / 8.0)
    return kc * (1.0 - fm) + km * fm


# ═══════════════════════════════════════════════════════════════════
# Neural network architectures
# ═══════════════════════════════════════════════════════════════════
class TempPINN(nn.Module):
    """Full-field PINN: (r̃, z̃, t̃) → T̂  (8 × 64, tanh)."""
    def __init__(self):
        super().__init__()
        dims = [3, 64, 64, 64, 64, 64, 64, 64, 64, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AblSurrogate(nn.Module):
    """1-D surrogate: Ep_norm → D_abl (µm)  (4 × 32, tanh)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32,  1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════
def to_t(arr, req_grad=False):
    return torch.tensor(np.asarray(arr, dtype=np.float32),
                        device=DEVICE, requires_grad=req_grad)

def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)

def mare_np(pred, ref, t_min=T_AMB + 5.0):
    mask = ref > t_min
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(pred[mask] - ref[mask]) / (ref[mask] - T_AMB + 1.0))) * 100


# ═══════════════════════════════════════════════════════════════════
# PART 1 — Spatio-temporal PINN  T(r, z, t)
# ═══════════════════════════════════════════════════════════════════
def _load_snapshot_dataset(mat: str):
    """
    Returns normalized (r̃, z̃, t̃, T) arrays plus full FDM grids for
    validation at pulses [35, 50].
    """
    d = np.load(FDM_OUT / f'{mat}_snapshots.npz')
    r_ax = d['r'];  z_ax = d['z']
    Nr, Nz = len(r_ax), len(z_ax)
    Rg, Zg = np.meshgrid(r_ax, z_ax, indexing='ij')   # (Nr, Nz)

    rng = np.random.default_rng(0)

    def collect(pulse_keys, n_pts=5000):
        rows = []
        for k in pulse_keys:
            T_k = d[f'T_pulse{k}']          # (Nr, Nz)
            t_n = float(k) / N_PULSES
            r_f = (Rg.ravel() / R_MAX).astype(np.float32)
            z_f = (Zg.ravel() / H_FILM).astype(np.float32)
            t_f = np.full(Nr * Nz, t_n, dtype=np.float32)
            T_f = T_k.ravel().astype(np.float32)
            rows.append(np.column_stack([r_f, z_f, t_f, T_f]))
        arr = np.concatenate(rows, axis=0)
        idx = rng.choice(len(arr), min(len(arr), n_pts * len(pulse_keys)),
                         replace=False)
        return arr[idx]

    train_arr = collect([1, 5, 10, 20], n_pts=3000)
    val_arr   = collect([35, 50],       n_pts=4000)

    T_mean = float(np.mean(train_arr[:, 3]))
    T_std  = max(float(np.std(train_arr[:, 3])), 1.0)

    X_tr  = to_t(train_arr[:, :3])
    Tn_tr = to_t((train_arr[:, 3] - T_mean) / T_std)
    X_va  = to_t(val_arr[:, :3])
    Tn_va = to_t((val_arr[:, 3] - T_mean) / T_std)

    # collocation points for PDE (interior only)
    rng2  = np.random.default_rng(1)
    n_col = 8000
    rc = rng2.uniform(0.02, 0.98, n_col).astype(np.float32)
    zc = rng2.uniform(0.02, 0.98, n_col).astype(np.float32)
    tc = rng2.uniform(0.02, 1.00, n_col).astype(np.float32)
    X_col = to_t(np.column_stack([rc, zc, tc]))

    # BC: r=0 symmetry nodes
    n_bc = 1200
    z_bc = rng2.uniform(0.01, 0.99, n_bc).astype(np.float32)
    t_bc = rng2.uniform(0.02, 1.00, n_bc).astype(np.float32)
    X_sym = to_t(np.column_stack([np.zeros(n_bc, dtype=np.float32), z_bc, t_bc]))

    # BC: z=0 surface nodes
    r_sf = rng2.uniform(0.00, 1.00, n_bc).astype(np.float32)
    t_sf = rng2.uniform(0.02, 1.00, n_bc).astype(np.float32)
    X_surf = to_t(np.column_stack([r_sf, np.zeros(n_bc, dtype=np.float32), t_sf]))

    return (X_tr, Tn_tr, X_va, Tn_va, X_col, X_sym, X_surf,
            T_mean, T_std, r_ax, z_ax, d)


def _pde_residual(model, X_col, cp_fn, k_fn, rho, T_mean, T_std):
    """Heat equation residual in physical units; normalised output."""
    x = X_col.clone().requires_grad_(True)
    T_n = model(x)
    T   = T_n * T_std + T_mean           # physical °C

    grads = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
    dTdr_n = grads[:, 0]
    dTdz_n = grads[:, 1]
    dTdt_n = grads[:, 2]

    # chain-rule: d/dr = d/d(r̃) * 1/R_MAX
    dTdr = dTdr_n * T_std / R_MAX
    dTdz = dTdz_n * T_std / H_FILM
    # dt̃ = dt/(pulse_dt*N_PULSES): use reference time scale for normalisation
    # (we only need the ratio to be consistent, not the exact value)
    dTdt = dTdt_n * T_std   # keep in (T_std · t̃⁻¹) units

    # 2nd spatial derivatives
    d2Tdr2_n = torch.autograd.grad(dTdr_n.sum(), x, create_graph=True)[0][:, 0]
    d2Tdr2   = d2Tdr2_n * T_std / R_MAX ** 2

    d2Tdz2_n = torch.autograd.grad(dTdz_n.sum(), x, create_graph=True)[0][:, 1]
    d2Tdz2   = d2Tdz2_n * T_std / H_FILM ** 2

    r_phys = (x[:, 0].detach() * R_MAX).clamp(min=1e-7)
    lap    = d2Tdr2 + dTdr / r_phys + d2Tdz2   # ∇²T in physical units

    T_det = T.detach()
    kT    = k_fn(T_det)
    cpT   = cp_fn(T_det)

    # Dimensionless residual: (ρ cp dT/dt̃ - k ∇²T) / (ρ cp T_std)
    res = (rho * cpT * dTdt - kT * lap) / (rho * cpT * T_std + 1e-6)
    return torch.mean(res ** 2)


def _bc_symmetry(model, X_sym):
    """Enforce ∂T̂/∂r̃ = 0 at r̃=0."""
    x = X_sym.clone().requires_grad_(True)
    Tn = model(x)
    g  = torch.autograd.grad(Tn.sum(), x, create_graph=True)[0]
    return torch.mean(g[:, 0] ** 2)


def _bc_surface(model, X_surf, T_mean, T_std, k_fn):
    """Enforce -k ∂T/∂z|_{z=0} = h(T-T∞) + εσ(T_K⁴-T∞_K⁴)."""
    x  = X_surf.clone().requires_grad_(True)
    Tn = model(x)
    T  = Tn * T_std + T_mean
    dTdz_n = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 1]
    dTdz   = dTdz_n.detach() * T_std / H_FILM

    T_det = T.detach()
    kT    = k_fn(T_det)
    TsK   = (T_det + 273.15).clamp(min=273.0)
    TaK   = torch.tensor(T_AMB + 273.15)
    q_bc  = H_CONV * (T_det - T_AMB) + EMISS * SB * (TsK ** 4 - TaK ** 4)
    q_scale = H_CONV * (T_ABL - T_AMB)       # ~8 kW/m²
    res   = (-kT * dTdz - q_bc) / (q_scale + 1.0)
    return torch.mean(res ** 2)


def train_temporal_pinn(mat: str = 'PP',
                        n_epochs: int = 6000,
                        lr: float = 1.5e-3,
                        lam_pde: float = 0.03,
                        lam_bc:  float = 0.005):
    print(f"\n{'='*60}")
    print(f"  Training temporal PINN [{mat}]  "
          f"epochs={n_epochs}  λ_pde={lam_pde}  λ_bc={lam_bc}")
    print(f"{'='*60}")

    rho   = RHO_PP   if mat == 'PP' else RHO_PE
    cp_fn = cp_pp_t  if mat == 'PP' else cp_pe_t
    k_fn  = k_pp_t   if mat == 'PP' else k_pe_t

    (X_tr, Tn_tr, X_va, Tn_va, X_col, X_sym, X_surf,
     T_mean, T_std, r_ax, z_ax, d_np) = _load_snapshot_dataset(mat)

    model = TempPINN().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    best_val, best_state = 1e9, None
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        loss_d   = mse(model(X_tr), Tn_tr)
        loss_pde = _pde_residual(model, X_col, cp_fn, k_fn, rho, T_mean, T_std)
        loss_bcs = _bc_symmetry(model, X_sym) + _bc_surface(model, X_surf, T_mean, T_std, k_fn)

        loss = loss_d + lam_pde * loss_pde + lam_bc * loss_bcs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        opt.step()
        sched.step()

        if ep % 500 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                Tn_pred_va = model(X_va)
                val_rmse   = float(mse(Tn_pred_va, Tn_va).sqrt())
            if val_rmse < best_val:
                best_val   = val_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elapsed = time.time() - t0
            print(f"  ep={ep:5d}  L_d={float(loss_d):.3e}  "
                  f"L_pde={float(loss_pde):.3e}  "
                  f"val_RMSE_n={val_rmse:.4f}  [{elapsed:.0f}s]")

    # restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # ── Full-grid predictions at pulse 35 and 50 ─────────────────
    model.eval()
    Nr, Nz = len(r_ax), len(z_ax)
    Rg, Zg = np.meshgrid(r_ax, z_ax, indexing='ij')
    r_f = (Rg.ravel() / R_MAX).astype(np.float32)
    z_f = (Zg.ravel() / H_FILM).astype(np.float32)

    def predict_grid(pulse_no):
        t_f = np.full_like(r_f, float(pulse_no) / N_PULSES)
        X   = to_t(np.column_stack([r_f, z_f, t_f]))
        with torch.no_grad():
            T_flat = model(X).numpy() * T_std + T_mean
        return T_flat.reshape(Nr, Nz).astype(np.float32)

    T_pinn_50 = predict_grid(50);  T_fdm_50 = d_np['T_pulse50'].astype(np.float32)
    T_pinn_35 = predict_grid(35);  T_fdm_35 = d_np['T_pulse35'].astype(np.float32)
    T_pinn_20 = predict_grid(20);  T_fdm_20 = d_np['T_pulse20'].astype(np.float32)

    mare_50 = mare_np(T_pinn_50, T_fdm_50)
    mare_35 = mare_np(T_pinn_35, T_fdm_35)
    mare_tr = mare_np(T_pinn_20, T_fdm_20)

    print(f"\n  MARE train-time (p20):  {mare_tr:.4f}%")
    print(f"  MARE val p35 :          {mare_35:.4f}%")
    print(f"  MARE val p50 :          {mare_50:.4f}%")

    # ── Per-pulse PINN predictions (for Fig 4) ─────────────────
    pulse_pts = [int(p) for p in d_np['pulses']]
    pinn_haz = []
    for p in pulse_pts:
        T_g = predict_grid(p)
        # HAZ radius: furthest r where T_max(z) > T_melt
        T_melt = T_M_PP if mat == 'PP' else T_M_PE
        haz_col = T_g.max(axis=1) > T_melt       # (Nr,) bool
        haz_r   = float(r_ax[haz_col][-1]) * 1e6 if haz_col.any() else 0.0
        pinn_haz.append(haz_r)
    pinn_haz = np.array(pinn_haz, dtype=np.float32)

    return {
        'model':      model,
        'r':          r_ax, 'z': z_ax,
        'T_fdm_50':   T_fdm_50, 'T_pinn_50': T_pinn_50,
        'T_fdm_35':   T_fdm_35, 'T_pinn_35': T_pinn_35,
        'mare_50':    mare_50,  'mare_35': mare_35,
        'T_mean':     T_mean,   'T_std':   T_std,
        'pinn_haz':   pinn_haz,
        'pulse_pts':  pulse_pts,
    }


# ═══════════════════════════════════════════════════════════════════
# PART 2 — Ablation-diameter surrogates  Ep → D_abl
# ═══════════════════════════════════════════════════════════════════
def _train_one_surrogate(EP_tr, D_tr, EP_all, label='ANN',
                          lam_mono=0.0, lam_concave=0.0,
                          n_epochs=4000, lr=1e-3):
    model = AblSurrogate()
    opt   = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)

    for ep in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        pred   = model(EP_tr)
        loss_d = mse(pred, D_tr)
        loss_p = torch.tensor(0.0)

        if lam_mono > 0 or lam_concave > 0:
            ep_col = torch.linspace(0.05, 0.98, 300, requires_grad=True).reshape(-1, 1)
            D_col  = model(ep_col)
            dD     = torch.autograd.grad(D_col.sum(), ep_col, create_graph=True)[0].squeeze()

            # monotonicity: D must not decrease with Ep
            loss_p = loss_p + lam_mono * torch.mean(torch.relu(-dD) ** 2)

            if lam_concave > 0:
                d2D = torch.autograd.grad(dD.sum(), ep_col,
                                           create_graph=True)[0].squeeze()
                # concavity (d²D/dEp² ≤ 0): penalise positive curvature
                loss_p = loss_p + lam_concave * torch.mean(torch.relu(d2D) ** 2)

        loss = loss_d + loss_p
        loss.backward()
        opt.step()
        sched.step()

        if ep % 1000 == 0 or ep == n_epochs:
            with torch.no_grad():
                rmse_tr = float(mse(model(EP_tr), D_tr).sqrt())
            print(f"  [{label}] ep={ep:4d}  loss={float(loss_d):.4e}  rmse={rmse_tr:.2f}µm")

    model.eval()
    with torch.no_grad():
        D_pred = model(EP_all).numpy().ravel()
    return model, D_pred


def train_ablation_surrogates(mat='PP', n_epochs=4000,
                               EP_NORM=300.0, EP_TRAIN_MAX=150.0):
    print(f"\n{'='*60}")
    print(f"  Training ablation surrogates [{mat}]  "
          f"train Ep ≤ {EP_TRAIN_MAX} µJ")
    print(f"{'='*60}")

    df   = pd.read_csv(FDM_OUT / f'{mat}_multiEp.csv')
    EP   = df['Ep_uJ'].values.astype(np.float32)
    DIAM = df['abl_dia_um'].values.astype(np.float32)

    ep_n      = (EP / EP_NORM).reshape(-1, 1)
    mask_tr   = EP <= EP_TRAIN_MAX
    EP_tr_t   = to_t(ep_n[mask_tr])
    D_tr_t    = to_t(DIAM[mask_tr])
    EP_all_t  = to_t(ep_n)

    pinn_m, D_pinn = _train_one_surrogate(
        EP_tr_t, D_tr_t, EP_all_t,
        label='PINN', lam_mono=0.8, lam_concave=0.5,
        n_epochs=n_epochs)

    ann_m, D_ann = _train_one_surrogate(
        EP_tr_t, D_tr_t, EP_all_t,
        label='ANN ', lam_mono=0.0, lam_concave=0.0,
        n_epochs=n_epochs)

    # metrics in test range
    mask_te = EP > EP_TRAIN_MAX
    if mask_te.any():
        mae_p = float(np.mean(np.abs(D_pinn[mask_te] - DIAM[mask_te])))
        mae_a = float(np.mean(np.abs(D_ann[mask_te]  - DIAM[mask_te])))
        print(f"\n  Test MAE (Ep>{EP_TRAIN_MAX}µJ): PINN={mae_p:.2f}µm  ANN={mae_a:.2f}µm")

    return {'EP': EP, 'DIAM_fdm': DIAM,
            'D_pinn': D_pinn, 'D_ann': D_ann,
            'EP_train_max': EP_TRAIN_MAX,
            'pinn_model': pinn_m}


# ═══════════════════════════════════════════════════════════════════
# PART 3 — Inverse design via gradient descent through PINN [Fig 5]
# ═══════════════════════════════════════════════════════════════════
def run_inverse_design(pinn_model: nn.Module, EP_NORM=300.0):
    """
    Use the trained PINN ablation surrogate as differentiable forward model.
    Optimise Ep to match three target diameters via Adam gradient descent.
    Returns per-iteration loss history + convergence iteration.
    """
    print(f"\n{'='*60}")
    print("  Inverse design: gradient descent through PINN surrogate")
    print(f"{'='*60}")

    targets = [65.0, 75.0, 85.0]     # target diameters (µm)
    conv_thresh  = 1e-3
    MAX_ITER     = 300

    loss_histories = []
    conv_iters     = []
    Ep_opt_list    = []
    D_achieved     = []

    for D_tgt in targets:
        Ep_n = torch.tensor([[0.35]], dtype=torch.float32, requires_grad=True)
        opt  = optim.Adam([Ep_n], lr=6e-3)
        hist = []
        conv_it = MAX_ITER

        for it in range(1, MAX_ITER + 1):
            opt.zero_grad()
            Ep_clamped = Ep_n.clamp(0.05, 0.95)
            D_pred     = pinn_model(Ep_clamped)
            loss       = ((D_pred - D_tgt) / D_tgt) ** 2
            loss.backward()
            opt.step()

            hist.append(float(loss.detach()))
            if float(loss.detach()) < conv_thresh and conv_it == MAX_ITER:
                conv_it = it

        Ep_opt  = float(Ep_n.clamp(0.05, 0.95).detach()) * EP_NORM
        D_final = float(pinn_model(Ep_n.clamp(0.05, 0.95).detach()))

        loss_histories.append(np.array(hist, dtype=np.float32))
        conv_iters.append(conv_it)
        Ep_opt_list.append(Ep_opt)
        D_achieved.append(D_final)

        print(f"  D*={D_tgt:.0f}µm → Ep*={Ep_opt:.1f}µJ  "
              f"D_achieved={D_final:.1f}µm  conv@it={conv_it}")

    return {
        'targets':    np.array(targets, dtype=np.float32),
        'loss_hist':  loss_histories,          # list of 3 arrays (300,)
        'conv_iters': np.array(conv_iters, dtype=np.int32),
        'Ep_opt':     np.array(Ep_opt_list, dtype=np.float32),
        'D_achieved': np.array(D_achieved, dtype=np.float32),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()

    # ── 1. Temporal PINN (PP + PE) ─────────────────────────────────
    res_pp = train_temporal_pinn('PP', n_epochs=6000)
    res_pe = train_temporal_pinn('PE', n_epochs=6000)

    # ── 2. Ablation surrogates (PP + PE) ───────────────────────────
    abl_pp = train_ablation_surrogates('PP', n_epochs=4000)
    abl_pe = train_ablation_surrogates('PE', n_epochs=4000)

    # ── 3. Inverse design ──────────────────────────────────────────
    inv = run_inverse_design(abl_pp['pinn_model'])

    # ── 4. Load FDM per-pulse metrics ──────────────────────────────
    d_pp = np.load(FDM_OUT / 'PP_snapshots.npz')
    d_pe = np.load(FDM_OUT / 'PE_snapshots.npz')

    # ── 5. Save all results ────────────────────────────────────────
    print("\n=== Saving fdm_output/pinn_results.npz ===")

    np.savez_compressed(
        FDM_OUT / 'pinn_results.npz',

        # ── Fig 3: temperature field validation
        pp_r           = res_pp['r'],
        pp_z           = res_pp['z'],
        pp_T_fdm_50    = res_pp['T_fdm_50'],
        pp_T_pinn_50   = res_pp['T_pinn_50'],
        pp_T_fdm_35    = res_pp['T_fdm_35'],
        pp_T_pinn_35   = res_pp['T_pinn_35'],
        pp_mare_50     = np.array([res_pp['mare_50']]),
        pp_mare_35     = np.array([res_pp['mare_35']]),

        pe_r           = res_pe['r'],
        pe_z           = res_pe['z'],
        pe_T_fdm_50    = res_pe['T_fdm_50'],
        pe_T_pinn_50   = res_pe['T_pinn_50'],
        pe_mare_50     = np.array([res_pe['mare_50']]),

        # ── Fig 4: per-pulse HAZ (FDM + PINN)
        pp_pulses      = d_pp['pulses'],
        pp_haz_fdm     = d_pp['haz_hist'],
        pp_rim_fdm     = d_pp['rim_hist'],
        pp_depth_fdm   = d_pp['depth_hist'],
        pp_haz_pinn    = res_pp['pinn_haz'],

        pe_pulses      = d_pe['pulses'],
        pe_haz_fdm     = d_pe['haz_hist'],
        pe_rim_fdm     = d_pe['rim_hist'],
        pe_depth_fdm   = d_pe['depth_hist'],
        pe_haz_pinn    = res_pe['pinn_haz'],

        # ── Fig 5: inverse design
        inv_targets    = inv['targets'],
        inv_loss_case1 = inv['loss_hist'][0],
        inv_loss_case2 = inv['loss_hist'][1],
        inv_loss_case3 = inv['loss_hist'][2],
        inv_conv_iters = inv['conv_iters'],
        inv_Ep_opt     = inv['Ep_opt'],
        inv_D_achieved = inv['D_achieved'],

        # ── Fig 6: ablation diameter surrogate
        pp_EP          = abl_pp['EP'],
        pp_DIAM_fdm    = abl_pp['DIAM_fdm'],
        pp_DIAM_pinn   = abl_pp['D_pinn'],
        pp_DIAM_ann    = abl_pp['D_ann'],
        pp_EP_train_max= np.array([abl_pp['EP_train_max']]),

        pe_EP          = abl_pe['EP'],
        pe_DIAM_fdm    = abl_pe['DIAM_fdm'],
        pe_DIAM_pinn   = abl_pe['D_pinn'],
        pe_DIAM_ann    = abl_pe['D_ann'],
    )

    elapsed = time.time() - t_total
    print(f"\n  Saved pinn_results.npz")
    print(f"  Total elapsed : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"\n  PP MARE (p50) : {res_pp['mare_50']:.4f}%  |  (p35): {res_pp['mare_35']:.4f}%")
    print(f"  PE MARE (p50) : {res_pe['mare_50']:.4f}%  |  (p35): {res_pe['mare_35']:.4f}%")
    print(f"\n  Inverse design convergence iters : {inv['conv_iters'].tolist()}")
    print(f"  Inverse optimal Ep               : {inv['Ep_opt'].tolist()} µJ")


if __name__ == '__main__':
    main()
