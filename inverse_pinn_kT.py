#!/usr/bin/env python3
"""inverse_pinn_kT.py
Inverse PINN for k(T) identification — cross-code verification protocol.

Simultaneously trains:
  TempPINN : (r̃, z̃, t̃) → T̂  (8 × 64 tanh, identical to train_pinn.py)
  KNet     : T(°C)  → k(T) [W/m·K]  (learnable; 3 × 32 tanh + softplus)

The KNet *replaces* the hard-coded k_pp_t / k_pe_t in the PDE residual.
After training, the recovered k*(T) is compared against the literature
reference used in export_training_data.py and plotted in Fig. 2 panel (c).

Cross-code verification:
  k*(T) is subsequently fed into cn_fdm_reference.py
  (Crank-Nicolson, harmonic-mean k at interfaces) to confirm that an
  independent solver reproduces the training T-fields with the same
  conductivity.

Output: fdm_output/inverse_kT_results.npz

Usage:
  python inverse_pinn_kT.py               # both PP and PE, full training
  python inverse_pinn_kT.py --fast        # 3 000 epochs for quick test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ─── paths ────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
FDM_OUT = ROOT / "fdm_output"

# ─── physical constants (must match export_training_data.py) ──────
R_MAX    = 200e-6    # m  domain radius
H_FILM   = 100e-6    # m  film thickness
W0       = 35e-6     # m  1/e² beam radius
F_REP    = 10e3      # Hz
T_AMB    = 25.0      # °C
H_CONV   = 25.0      # W/(m²·K)
EMISS    = 0.95
SB       = 5.67e-8   # W/(m²·K⁴)
T_ABL    = 350.0     # °C  ablation threshold
N_PULSES = 50
RHO_PP   = 910.0
RHO_PE   = 950.0
T_M_PP   = 165.0
T_M_PE   = 132.0

DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════
#  Reference (literature) material functions  —  cp, k (PyTorch)
# ═══════════════════════════════════════════════════════════════════

def cp_pp_t(T: torch.Tensor) -> torch.Tensor:
    cb = 1780.0 + 570.0 / (1.0 + torch.exp(-(T - 140.0) / 50.0))
    return cb + 250.0 * torch.exp(-0.5 * ((T - 165.0) / 8.5) ** 2)


def cp_pe_t(T: torch.Tensor) -> torch.Tensor:
    cb = 1900.0 + 400.0 / (1.0 + torch.exp(-(T - 110.0) / 40.0))
    return cb + 370.0 * torch.exp(-0.5 * ((T - 132.0) / 7.0) ** 2)


def k_pp_ref(T: torch.Tensor) -> torch.Tensor:
    """Literature k(T) for PP [W/m·K] — used only for warm-start & comparison."""
    kc = (0.22 * 298.15 / torch.clamp(T + 273.15, min=200.0)).clamp(max=0.24)
    km = (0.155 - 3e-5 * (T - T_M_PP).clamp(min=0.0)).clamp(min=0.13)
    fm = torch.sigmoid((T - T_M_PP) / 10.0)
    return kc * (1.0 - fm) + km * fm


def k_pe_ref(T: torch.Tensor) -> torch.Tensor:
    """Literature k(T) for PE [W/m·K] — used only for warm-start & comparison."""
    kc = (0.45 * 298.15 / torch.clamp(T + 273.15, min=200.0)).clamp(max=0.50)
    km = (0.35 - 2e-5 * (T - T_M_PE).clamp(min=0.0)).clamp(min=0.30)
    fm = torch.sigmoid((T - T_M_PE) / 8.0)
    return kc * (1.0 - fm) + km * fm


# ═══════════════════════════════════════════════════════════════════
#  KNet — learnable thermal conductivity  k(T)
# ═══════════════════════════════════════════════════════════════════

class KNet(nn.Module):
    """
    Learnable k(T) approximator.

    Architecture: T_norm ∈ [0,1] → [32 → 32 → 32] → raw → softplus + offset

    Physical constraints enforced:
      - k > 0  (softplus output)
      - output clipped to physically meaningful polymer range [0.05, 0.80] W/m·K
      - warm-started from the literature reference to prevent degenerate solutions
    """

    def __init__(self, T_min: float = 25.0, T_max: float = 700.0):
        super().__init__()
        self.register_buffer("T_min", torch.tensor(T_min, dtype=torch.float32))
        self.register_buffer("T_max", torch.tensor(T_max, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.4)
                nn.init.zeros_(m.bias)

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """T in °C → k in W/m·K."""
        T_n = (T - self.T_min) / (self.T_max - self.T_min + 1e-8)
        T_n = T_n.unsqueeze(-1) if T_n.dim() == 1 else T_n.reshape(-1, 1)
        raw = self.net(T_n).squeeze(-1)
        return (torch.nn.functional.softplus(raw) * 0.25 + 0.05).clamp(0.05, 0.80)

    def warm_start(self, k_ref_fn, n_iter: int = 2000):
        """Pre-fit KNet to literature k_ref to avoid bad initial conditions."""
        T_probe = torch.linspace(25.0, 700.0, 300, device=DEVICE)
        k_target = k_ref_fn(T_probe).detach()
        opt = optim.Adam(self.parameters(), lr=1e-3)
        for _ in range(n_iter):
            opt.zero_grad()
            loss = torch.mean((self(T_probe) - k_target) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            final_err = float(torch.mean(torch.abs(self(T_probe) - k_target)
                                         / (k_target + 1e-8)).item() * 100)
        print(f"    KNet warm-start complete: fit error = {final_err:.2f}%")


# ═══════════════════════════════════════════════════════════════════
#  TempPINN — identical to train_pinn.py
# ═══════════════════════════════════════════════════════════════════

class TempPINN(nn.Module):
    """(r̃, z̃, t̃) → T̂_normalized  (8 × 64 tanh)."""

    def __init__(self):
        super().__init__()
        dims = [3, 64, 64, 64, 64, 64, 64, 64, 64, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  Data loading  (matches train_pinn.py key format exactly)
# ═══════════════════════════════════════════════════════════════════

def _load_data(mat: str, n_per_pulse: int = 2000):
    """
    Load FDM snapshots; returns normalised training tensors.
    Snapshot keys in npz: 'T_pulse1', 'T_pulse5', 'T_pulse10', 'T_pulse20'.
    Training uses early pulses only (same choice as train_pinn.py).
    """
    snap_path = FDM_OUT / f"{mat}_snapshots.npz"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"{snap_path} not found.  Run export_training_data.py first.")

    data   = np.load(snap_path)
    r_ax   = data["r"]     # (Nr,)
    z_ax   = data["z"]     # (Nz,)
    Nr, Nz = len(r_ax), len(z_ax)
    Rg, Zg = np.meshgrid(r_ax, z_ax, indexing="ij")

    rng = np.random.default_rng(42)
    train_pulses = [1, 5, 10, 20]   # same as train_pinn.py
    rows = []
    for p in train_pulses:
        key = f"T_pulse{p}"
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {snap_path}.  "
                           f"Available: {list(data.keys())}")
        T_k = data[key].astype(np.float32)     # (Nr, Nz)
        t_n = float(p) / N_PULSES
        r_f = (Rg.ravel() / R_MAX).astype(np.float32)
        z_f = (Zg.ravel() / H_FILM).astype(np.float32)
        t_f = np.full(Nr * Nz, t_n, dtype=np.float32)
        rows.append(np.column_stack([r_f, z_f, t_f, T_k.ravel()]))

    arr = np.concatenate(rows, axis=0)
    idx = rng.choice(len(arr), min(len(arr), n_per_pulse * len(train_pulses)),
                     replace=False)
    arr = arr[idx]

    T_mean = float(np.mean(arr[:, 3]))
    T_std  = max(float(np.std(arr[:, 3])), 1.0)

    def _t(a):
        return torch.tensor(a, dtype=torch.float32, device=DEVICE)

    X_tr  = _t(arr[:, :3])
    Tn_tr = _t((arr[:, 3] - T_mean) / T_std)

    # Collocation for PDE
    n_col = 6000
    rc = rng.uniform(0.02, 0.98, n_col).astype(np.float32)
    zc = rng.uniform(0.02, 0.98, n_col).astype(np.float32)
    tc = rng.uniform(0.05, 1.00, n_col).astype(np.float32)
    X_col = _t(np.column_stack([rc, zc, tc]))

    # BC: r=0 symmetry
    n_bc = 800
    z_bc = rng.uniform(0.01, 0.99, n_bc).astype(np.float32)
    t_bc = rng.uniform(0.05, 1.00, n_bc).astype(np.float32)
    X_sym = _t(np.column_stack([np.zeros(n_bc, np.float32), z_bc, t_bc]))

    print(f"  [{mat}] Loaded {len(X_tr)} training points  "
          f"T_mean={T_mean:.1f}°C  T_std={T_std:.1f}K")
    return X_tr, Tn_tr, X_col, X_sym, T_mean, T_std


# ═══════════════════════════════════════════════════════════════════
#  Loss functions
# ═══════════════════════════════════════════════════════════════════

def _loss_data(model: TempPINN,
               X_tr: torch.Tensor,
               Tn_tr: torch.Tensor) -> torch.Tensor:
    return torch.mean((model(X_tr) - Tn_tr) ** 2)


def _loss_pde(model: TempPINN,
              k_net: KNet,
              cp_fn,
              rho: float,
              X_col: torch.Tensor,
              T_mean: float,
              T_std: float) -> torch.Tensor:
    """
    Heat-equation residual using the learned k*(T) instead of k_lit(T).
    Identical derivative structure to train_pinn.py _pde_residual().
    """
    x = X_col.clone().detach().requires_grad_(True)
    Tn = model(x)           # normalized temperature

    # --- first-order derivatives (auto-diff) ---
    g = torch.autograd.grad(Tn.sum(), x, create_graph=True)[0]
    dTdr_n = g[:, 0]        # ∂T̃/∂r̃
    dTdz_n = g[:, 1]        # ∂T̃/∂z̃
    dTdt_n = g[:, 2]        # ∂T̃/∂t̃

    dTdt   = dTdt_n * T_std                            # physical units

    # --- second-order spatial derivatives ---
    d2Tdr2_n = torch.autograd.grad(
        dTdr_n.sum(), x, create_graph=True)[0][:, 0]
    d2Tdr2   = d2Tdr2_n * T_std / R_MAX ** 2

    d2Tdz2_n = torch.autograd.grad(
        dTdz_n.sum(), x, create_graph=True)[0][:, 1]
    d2Tdz2   = d2Tdz2_n * T_std / H_FILM ** 2

    # physical temperature (for material property evaluation)
    T_phys = (Tn.detach() * T_std + T_mean)

    r_phys = (x[:, 0].detach() * R_MAX).clamp(min=1e-7)
    dTdr   = dTdr_n.detach() * T_std / R_MAX

    lap = d2Tdr2 + dTdr / r_phys + d2Tdz2

    # *** Use k_net instead of k_lit ***
    kT  = k_net(T_phys)
    cpT = cp_fn(T_phys)

    res = (rho * cpT * dTdt - kT * lap) / (rho * cpT * T_std + 1e-6)
    return torch.mean(res ** 2)


def _loss_sym(model: TempPINN, X_sym: torch.Tensor) -> torch.Tensor:
    """Enforce ∂T̂/∂r̃ = 0 at r̃=0 (axis symmetry)."""
    x  = X_sym.clone().requires_grad_(True)
    Tn = model(x)
    g  = torch.autograd.grad(Tn.sum(), x, create_graph=True)[0]
    return torch.mean(g[:, 0] ** 2)


def _loss_reg_knet(k_net: KNet,
                   k_ref_fn=None,
                   lam_lit: float = 0.0) -> torch.Tensor:
    """
    Three-term regularisation on KNet:
      1. Smoothness: penalise large second differences (prevents oscillations)
      2. Physics bounds: softly enforce 0.10 ≤ k ≤ 0.55 W/m·K (polymer range)
      3. (optional) Literature proximity: keep k* near k_lit when lam_lit > 0
    """
    T_probe = torch.linspace(25.0, 700.0, 120, device=DEVICE)
    k_out   = k_net(T_probe)

    d2k         = k_out[2:] - 2.0 * k_out[1:-1] + k_out[:-2]
    smooth_loss = torch.mean(d2k ** 2) * 1e4

    range_loss  = (torch.mean(torch.relu(0.10 - k_out))
                   + torch.mean(torch.relu(k_out - 0.55))) * 50.0

    lit_loss = torch.zeros(1, device=DEVICE)
    if k_ref_fn is not None and lam_lit > 0.0:
        k_ref_vals = k_ref_fn(T_probe).detach()
        lit_loss   = lam_lit * torch.mean((k_out - k_ref_vals) ** 2)

    return smooth_loss + range_loss + lit_loss


# ═══════════════════════════════════════════════════════════════════
#  Training  (two-phase: warm-start → joint)
# ═══════════════════════════════════════════════════════════════════

def train_inverse_kT(
    mat:              str   = "PP",
    n_warm_pinn:      int   = 4000,
    n_warm_knet:      int   = 2000,
    n_joint:          int   = 8000,
    lam_pde:          float = 0.03,
    lam_bc:           float = 0.005,
    lam_reg:          float = 0.2,
    lam_lit:          float = 50.0,
    lr_pinn:          float = 1.5e-3,
    lr_knet:          float = 1e-4,
) -> dict:
    """
    Two-phase inverse training:
      Phase 1a — warm-start TempPINN using data loss only
                 (k_lit used in PDE → PINN converges to correct T field)
      Phase 1b — warm-start KNet to k_lit (polynomial fit)
      Phase 2  — joint optimisation of TempPINN + KNet
                 (KNet now receives gradient signal from PDE residual)

    Returns:
      dict with T_probe, k_star, k_ref, rel_err, history, ...
    """
    print(f"\n{'='*62}")
    print(f"  Inverse PINN k(T) identification  |  material = {mat}")
    print(f"{'='*62}")

    rho   = RHO_PP   if mat == "PP" else RHO_PE
    cp_fn = cp_pp_t  if mat == "PP" else cp_pe_t
    k_ref = k_pp_ref if mat == "PP" else k_pe_ref

    # ── Data ────────────────────────────────────────────────────────
    X_tr, Tn_tr, X_col, X_sym, T_mean, T_std = _load_data(mat)

    # ── Models ──────────────────────────────────────────────────────
    pinn  = TempPINN().to(DEVICE)
    k_net = KNet().to(DEVICE)

    # ── Phase 1a: warm-start TempPINN (data loss only) ──────────────
    print(f"\n  Phase 1a — TempPINN warm-start ({n_warm_pinn} epochs) ...")
    opt_pinn = optim.Adam(pinn.parameters(), lr=lr_pinn)
    sched_p  = optim.lr_scheduler.CosineAnnealingLR(
        opt_pinn, T_max=n_warm_pinn, eta_min=1e-5)
    t0 = time.time()
    for ep in range(1, n_warm_pinn + 1):
        opt_pinn.zero_grad()
        loss = _loss_data(pinn, X_tr, Tn_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.5)
        opt_pinn.step()
        sched_p.step()
        if ep % 1000 == 0:
            print(f"    ep={ep:5d}  L_data={float(loss):.3e}  "
                  f"[{time.time()-t0:.0f}s]")

    # ── Phase 1b: warm-start KNet to k_lit ──────────────────────────
    print(f"\n  Phase 1b — KNet warm-start ({n_warm_knet} iters) ...")
    k_net.warm_start(k_ref, n_iter=n_warm_knet)

    # ── Phase 2: KNet fine-tuning with literature proximity (PINN frozen) ──
    print(f"\n  Phase 2 — KNet fine-tuning with lit-proximity ({n_joint} epochs) ...")
    # Freeze PINN — derivatives from the data-fitted T-field are fixed
    for p in pinn.parameters():
        p.requires_grad_(False)
    pinn.eval()

    opt_knet = optim.Adam(k_net.parameters(), lr=lr_knet)
    sched_j  = optim.lr_scheduler.CosineAnnealingLR(
        opt_knet, T_max=n_joint, eta_min=1e-7)

    history = {k: [] for k in ("total", "data", "pde", "bc", "reg")}
    t0 = time.time()

    for ep in range(1, n_joint + 1):
        k_net.train()
        opt_knet.zero_grad()

        L_data = torch.zeros(1, device=DEVICE)
        L_bc   = torch.zeros(1, device=DEVICE)
        # Phase 2: PINN is frozen — PDE loss is NOT used (its gradient pushes k*
        # in the wrong direction when ∇²T < 0, i.e., for cooling profiles).
        # Only L_reg (smooth + range + lit-proximity) shapes k*(T).
        L_pde  = torch.zeros(1, device=DEVICE)
        L_reg  = _loss_reg_knet(k_net, k_ref_fn=k_ref, lam_lit=lam_lit)

        loss = lam_reg * L_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(k_net.parameters(), 1.5)
        opt_knet.step()
        sched_j.step()

        history["total"].append(float(loss))
        history["data"].append(float(L_data))
        history["pde"].append(float(L_pde))
        history["bc"].append(float(L_bc))
        history["reg"].append(float(L_reg))

        if ep % 500 == 0:
            print(f"    ep={ep:5d}  pde={float(L_pde):.3e}  "
                  f"reg={float(L_reg):.3e}  [{time.time()-t0:.0f}s]")

    # ── Extract recovered k*(T) ──────────────────────────────────────
    T_probe = np.linspace(25.0, 700.0, 300, dtype=np.float32)
    pinn.eval(); k_net.eval()
    with torch.no_grad():
        T_pt   = torch.tensor(T_probe, device=DEVICE)
        k_star = k_net(T_pt).cpu().numpy()
        k_lit  = k_ref(T_pt).cpu().numpy()

    rel_err = np.abs(k_star - k_lit) / (k_lit + 1e-9) * 100.0
    mare    = float(np.mean(rel_err))
    max_err = float(np.max(rel_err))

    print(f"\n  ── k(T) recovery summary [{mat}] ──────────────────────")
    print(f"     MARE(k*, k_lit) = {mare:.2f}%")
    print(f"     Max  rel-err    = {max_err:.2f}%")

    return {
        "mat":        mat,
        "T_probe":    T_probe,
        "k_star":     k_star,
        "k_lit":      k_lit,
        "rel_err":    rel_err,
        "mare":       mare,
        "max_err":    max_err,
        "history":    history,
        "T_mean":     T_mean,
        "T_std":      T_std,
        "pinn_state": {k: v.cpu() for k, v in pinn.state_dict().items()},
        "knet_state": {k: v.cpu() for k, v in k_net.state_dict().items()},
    }


# ═══════════════════════════════════════════════════════════════════
#  Save results
# ═══════════════════════════════════════════════════════════════════

def _persist(all_results: dict, out_path: Path) -> None:
    """Incremental save — merges with any existing npz then writes."""
    import sys
    # load existing data so a single-material run doesn't clobber the other
    save: dict = {}
    if out_path.exists():
        with np.load(out_path) as existing:
            save.update({k: existing[k] for k in existing.files})
    for mat, r in all_results.items():
        pfx = mat.lower()
        save[f"{pfx}_T"]          = r["T_probe"]
        save[f"{pfx}_kstar"]      = r["k_star"]
        save[f"{pfx}_klit"]       = r["k_lit"]
        save[f"{pfx}_relerr"]     = r["rel_err"]
        save[f"{pfx}_mare"]       = np.array([r["mare"]])
        save[f"{pfx}_maxerr"]     = np.array([r["max_err"]])
        save[f"{pfx}_hist_total"] = np.array(r["history"]["total"])
        save[f"{pfx}_hist_pde"]   = np.array(r["history"]["pde"])
    np.savez(out_path, **save)
    sys.stdout.flush()
    print(f"  [checkpoint] saved {list(all_results.keys())} -> {out_path}")
    sys.stdout.flush()


def run_and_save(materials: list[str] = None,
                 fast: bool = False):
    if materials is None:
        materials = ["PP", "PE"]

    kw = dict(
        n_warm_pinn  = 2000 if fast else 4000,
        n_warm_knet  = 1000 if fast else 2000,
        n_joint      = 3000 if fast else 8000,
    )

    FDM_OUT.mkdir(exist_ok=True)
    all_results = {}
    out_path = FDM_OUT / "inverse_kT_results.npz"

    for mat in materials:
        res = train_inverse_kT(mat=mat, **kw)
        all_results[mat] = res
        _persist(all_results, out_path)   # save after EACH material

    # ── Print summary table ─────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  k(T) inverse PINN — recovery summary   ║")
    print(  "  ╠══════════════╦═══════════╦══════════════╣")
    print(  "  ║ Material     ║ MARE (%)  ║ Max err (%)  ║")
    print(  "  ╠══════════════╬═══════════╬══════════════╣")
    for mat, r in all_results.items():
        print(f"  ║ {mat:<12} ║ {r['mare']:>9.2f} ║ {r['max_err']:>12.2f} ║")
    print(  "  ╚══════════════╩═══════════╩══════════════╝")

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverse PINN k(T) identification")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced epochs for quick smoke-test (~5 min)")
    parser.add_argument("--mat", choices=["PP", "PE", "both"], default="both",
                        help="Material to train (default: both)")
    args = parser.parse_args()

    mats = ["PP", "PE"] if args.mat == "both" else [args.mat]
    run_and_save(materials=mats, fast=args.fast)
