#!/usr/bin/env python3
"""cn_fdm_reference.py
Independent Crank-Nicolson (CN) FDM reference solver.

PURPOSE — cross-code verification of k*(T) recovered by inverse_pinn_kT.py.

This solver is *intentionally* different from export_training_data.py:
  ┌───────────────────────┬──────────────────────────────────────┐
  │ export_training_data  │ cn_fdm_reference  (this file)        │
  ├───────────────────────┼──────────────────────────────────────┤
  │ Explicit Forward-Euler│ Implicit Crank-Nicolson (z-direction)│
  │ Arithmetic mean k@ifc │ Harmonic mean k at interfaces         │
  │ Fixed time-step CFL   │ Unconditionally stable; larger dt    │
  │ Pure numpy stencil    │ scipy.linalg.solve_banded tridiag    │
  └───────────────────────┴──────────────────────────────────────┘

If  T_CN(k*) ≈ T_ppFDM(k_lit)  then k*(T) is independently validated.

Usage:
  from cn_fdm_reference import run_cn_fdm, compare_with_ppFDM
  result = run_cn_fdm('PP', k_func=my_k_fn)
  compare_with_ppFDM('PP', k_func=my_k_fn)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded

ROOT    = Path(__file__).resolve().parent.parent
FDM_OUT = ROOT / "fdm_output"

# ─── Physical constants (identical to export_training_data.py) ────
R_MAX   = 200e-6    # m  domain radius
H_FILM  = 100e-6    # m  film thickness
W0      = 35e-6     # m
F_REP   = 10e3      # Hz
T_AMB   = 25.0      # °C
H_CONV  = 25.0      # W/(m²·K)
EMISS   = 0.95
SB      = 5.67e-8

# Material parameters
_PARAMS = {
    "PP": dict(rho=910.0,  T_melt=165.0, T_abl=350.0, alpha_abs=0.6e5,
               L_melt=100e3, L_decomp=300e3),
    "PE": dict(rho=950.0,  T_melt=132.0, T_abl=350.0, alpha_abs=0.4e5,
               L_melt=200e3, L_decomp=280e3),
}


# ─── Reference cp(T) functions (must match export_training_data.py) ──

def _cp_pp(T: np.ndarray) -> np.ndarray:
    cb = 1780.0 + 570.0 / (1.0 + np.exp(-(T - 140.0) / 50.0))
    return cb + 250.0 * np.exp(-0.5 * ((T - 165.0) / 8.5) ** 2)


def _cp_pe(T: np.ndarray) -> np.ndarray:
    cb = 1900.0 + 400.0 / (1.0 + np.exp(-(T - 110.0) / 40.0))
    return cb + 370.0 * np.exp(-0.5 * ((T - 132.0) / 7.0) ** 2)


# ─── Reference k(T) functions (literature — used for comparison) ──

def k_pp_lit(T: np.ndarray) -> np.ndarray:
    kc = np.minimum(0.22 * 298.15 / np.maximum(T + 273.15, 200.0), 0.24)
    km = np.maximum(0.155 - 3e-5 * np.maximum(T - 165.0, 0.0), 0.13)
    fm = 1.0 / (1.0 + np.exp(-(T - 165.0) / 10.0))
    return kc * (1.0 - fm) + km * fm


def k_pe_lit(T: np.ndarray) -> np.ndarray:
    kc = np.minimum(0.45 * 298.15 / np.maximum(T + 273.15, 200.0), 0.50)
    km = np.maximum(0.35 - 2e-5 * np.maximum(T - 132.0, 0.0), 0.30)
    fm = 1.0 / (1.0 + np.exp(-(T - 132.0) / 8.0))
    return kc * (1.0 - fm) + km * fm


# ═══════════════════════════════════════════════════════════════════
#  Core utilities
# ═══════════════════════════════════════════════════════════════════

def _k_harmonic(T_a: np.ndarray,
                T_b: np.ndarray,
                k_func: Callable) -> np.ndarray:
    """
    Harmonic mean conductivity at the interface between two cells:
        k_face = 2 k(T_a) k(T_b) / [k(T_a) + k(T_b)]
    This is MORE accurate than arithmetic mean for problems where k varies
    strongly with T, and is the key numerical difference from export_training_data.py.
    """
    ka = k_func(T_a)
    kb = k_func(T_b)
    return 2.0 * ka * kb / np.clip(ka + kb, 1e-12, None)


# ═══════════════════════════════════════════════════════════════════
#  ADI Crank-Nicolson: z-implicit, r-explicit
#  (Different from explicit-FDM in export_training_data.py)
# ═══════════════════════════════════════════════════════════════════

def _cn_z_step(T: np.ndarray,
               dz: float,
               dt: float,
               rho: float,
               cp_fn: Callable,
               k_func: Callable,
               Q_half: np.ndarray) -> np.ndarray:
    """
    Half-step: implicit in z, explicit in r.

    For each radial strip i, solves the tridiagonal system:
        (ρ cₚ/dt) Tⁿ⁺¹ - (1/2) ∂_z[k ∂_z Tⁿ⁺¹]
            = (ρ cₚ/dt) Tⁿ + (1/2) ∂_z[k ∂_z Tⁿ] + Q_half + r-explicit

    r-explicit term computed outside this function and folded into Q_half.
    Boundary conditions:
      z=0 (surface):  -k ∂T/∂z = h_c (T - T_amb)   (Newton cooling)
      z=H (bottom):   -k ∂T/∂z = h_c (T - T_amb)   (symmetric bottom BC)
    """
    Nr, Nz = T.shape
    T_new = T.copy()

    for i in range(Nr):
        Tc   = T[i, :]           # current z-column
        cp   = cp_fn(Tc)
        rcp_dt = rho * cp / dt   # (Nz,)

        # Harmonic-mean conductivities at z-interfaces
        kp = np.zeros(Nz)        # k at z+1/2 interface
        km = np.zeros(Nz)        # k at z-1/2 interface
        kp[:-1] = _k_harmonic(Tc[:-1], Tc[1:],  k_func)
        kp[-1]  = k_func(Tc[-1])
        km[1:]  = _k_harmonic(Tc[:-1], Tc[1:],  k_func)
        km[0]   = k_func(Tc[0])

        # Build banded matrix  ab[0,:]=upper, ab[1,:]=diag, ab[2,:]=lower
        ab  = np.zeros((3, Nz))
        rhs = np.zeros(Nz)

        # Explicit z-diffusion contribution (Crank-Nicolson: 0.5 each side)
        cn_coef = 0.5 / dz ** 2
        for j in range(Nz):
            if j == 0:
                # z=0 surface: CN discretisation + Newton BC
                k_face = kp[0]
                ab[1, j] = rcp_dt[j] + cn_coef * k_face + H_CONV / dz
                ab[0, 1] = -cn_coef * k_face
                # explicit side
                d2T_ex = cn_coef * k_face * (Tc[1] - Tc[0]) - H_CONV * (Tc[0] - T_AMB) / dz
                rhs[j]  = rcp_dt[j] * Tc[j] + d2T_ex + Q_half[i, j]

            elif j == Nz - 1:
                # z=H bottom: Newton BC (symmetric)
                k_face = km[-1]
                ab[1, j]  = rcp_dt[j] + cn_coef * k_face + H_CONV / dz
                ab[2, j-1]= -cn_coef * k_face
                d2T_ex    = cn_coef * k_face * (Tc[-2] - Tc[-1]) - H_CONV * (Tc[-1] - T_AMB) / dz
                rhs[j]    = rcp_dt[j] * Tc[j] + d2T_ex + Q_half[i, j]

            else:
                ab[1, j]  = rcp_dt[j] + cn_coef * (kp[j] + km[j])
                ab[0, j+1]= -cn_coef * kp[j]
                ab[2, j-1]= -cn_coef * km[j]
                d2T_ex    = cn_coef * (kp[j] * (Tc[j+1] - Tc[j])
                                       - km[j] * (Tc[j] - Tc[j-1]))
                rhs[j]    = rcp_dt[j] * Tc[j] + d2T_ex + Q_half[i, j]

        T_new[i, :] = solve_banded((1, 1), ab, rhs)

    return T_new


def _explicit_r_laplacian(T: np.ndarray,
                           r: np.ndarray,
                           dr: float,
                           k_func: Callable) -> np.ndarray:
    """
    Explicit radial diffusion term: (1/r) ∂/∂r [r k ∂T/∂r]
    using harmonic-mean k at r-interfaces.
    Returns array of shape (Nr, Nz).
    """
    Nr, Nz = T.shape
    lap_r  = np.zeros_like(T)

    # Interior r nodes
    for i in range(1, Nr - 1):
        kip = _k_harmonic(T[i, :], T[i+1, :], k_func)   # k at r+1/2
        kim = _k_harmonic(T[i-1, :], T[i, :], k_func)   # k at r-1/2
        rip = r[i] + 0.5 * dr
        rim = r[i] - 0.5 * dr
        lap_r[i, :] = (kip * rip * (T[i+1, :] - T[i, :])
                        - kim * rim * (T[i, :] - T[i-1, :])) / (r[i] * dr ** 2)

    # r=0: L'Hopital limit → (1/r) ∂/∂r[r k ∂T/∂r]|_{r=0} = 2k ∂²T/∂r²
    k0 = k_func(T[0, :])
    lap_r[0, :] = 2.0 * k0 * (T[1, :] - T[0, :]) / dr ** 2

    # r=R_MAX: adiabatic (zero flux)
    lap_r[-1, :] = 0.0

    return lap_r


# ═══════════════════════════════════════════════════════════════════
#  Main CN-FDM solver
# ═══════════════════════════════════════════════════════════════════

def run_cn_fdm(
    material: str,
    k_func: Callable[[np.ndarray], np.ndarray],
    N: int = 60,
    N_pulses: int = 50,
    Ep: float = 100e-6,
    snapshot_pulses: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run 2-D axisymmetric Crank-Nicolson FDM with arbitrary k(T).

    Key differences from export_training_data.py (explicit FDM):
      • Implicit CN time stepping in z (unconditionally stable)
      • Harmonic mean k at all cell interfaces
      • scipy banded-matrix tridiagonal solver

    Args:
        material:        'PP' or 'PE'
        k_func:          callable: T[°C] np.ndarray → k[W/m·K] np.ndarray
                         Pass k_pp_lit / k_pe_lit for literature reference,
                         or a KNet.numpy_fn() for the recovered k*(T).
        N:               grid points per side (N×N grid)
        N_pulses:        number of laser pulses to simulate
        Ep:              pulse energy [J]
        snapshot_pulses: pulse numbers at which to save full T field

    Returns:
        dict with keys:
          r, z             — grid axes [m]
          snapshots        — {pulse_number: T_field (N×N) [°C]}
          haz_history      — HAZ radius [µm] per pulse
          depth_history    — ablation depth [µm] per pulse
          rim_history      — rim height [µm] per pulse
          pulse_range      — np.arange(1, N_pulses+1)
    """
    if snapshot_pulses is None:
        snapshot_pulses = [1, 5, 10, 20, 35, 50]

    prm = _PARAMS[material]
    rho  = prm["rho"];  T_abl = prm["T_abl"];  T_melt = prm["T_melt"]
    alpha_abs = prm["alpha_abs"]
    L_melt    = prm["L_melt"];   L_decomp = prm["L_decomp"]
    cp_fn = _cp_pp if material == "PP" else _cp_pe

    # ── Grid ─────────────────────────────────────────────────────
    r  = np.linspace(0.0, R_MAX, N)
    z  = np.linspace(0.0, H_FILM, N)
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    Rg, Zg = np.meshgrid(r, z, indexing="ij")   # (N, N)

    # ── Laser source — spatial part ──────────────────────────────
    tau_p = 10e-9    # pulse duration [s]
    T_rep = 1.0 / F_REP
    f_pk  = 2.0 * Ep / (np.pi * W0 ** 2)        # peak fluence [J/m²]
    Q_sp  = (f_pk * alpha_abs
              * np.exp(-2.0 * Rg ** 2 / W0 ** 2)
              * np.exp(-alpha_abs * Zg))          # (N,N) J/m³ per pulse

    # ── Time step: CN is unconditionally stable; choose dt for accuracy
    #    Use dt = T_rep / 100 (5× larger than CFL-limited explicit dt)
    dt     = T_rep / 100.0
    N_step = max(int(T_rep / dt), 10)
    N_on   = max(1, int(tau_p / dt))             # steps during pulse

    if verbose:
        k_test = float(k_func(np.array([T_AMB]))[0])
        alpha0 = k_test / (rho * float(cp_fn(np.array([T_AMB]))[0]))
        dt_cfl = 0.3 * min(dr, dz) ** 2 / alpha0
        print(f"  [{material}]  Grid {N}×{N}  dt={dt*1e6:.2f} µs  "
              f"CFL-limit≈{dt_cfl*1e6:.2f} µs  "
              f"(CN: {dt/dt_cfl:.1f}× larger dt OK)")

    # ── Initial condition ─────────────────────────────────────────
    T    = np.full((N, N), T_AMB, dtype=np.float64)
    abl  = np.zeros((N, N), dtype=bool)
    lat  = np.zeros((N, N), dtype=np.float64)

    snapshots    = {}
    haz_history  = []
    depth_history= []
    rim_history  = []

    for pulse_n in range(1, N_pulses + 1):

        # ── 1. Instantaneous heat deposition (Beer-Lambert) ──────
        front_j = np.where(
            np.any(~abl, axis=1),
            np.argmax(~abl, axis=1),
            N).astype(np.int32)
        z_front = np.where(front_j < N, z[np.minimum(front_j, N-1)], H_FILM)

        Z_rel  = np.maximum(Zg - z_front[:, np.newaxis], 0.0)
        Q_vol  = (f_pk * alpha_abs
                  * np.exp(-2.0 * Rg ** 2 / W0 ** 2)
                  * np.exp(-alpha_abs * Z_rel)
                  * (~abl))
        T      = T + Q_vol / (rho * cp_fn(T))

        # ── 2. Latent heat + ablation marking ────────────────────
        # melt
        mask_m = (~abl) & (T > T_melt) & (lat < L_melt)
        if np.any(mask_m):
            cp_m   = cp_fn(T)
            excess = rho * cp_m * (T - T_melt)
            need   = L_melt - lat
            lat    = np.where(mask_m,
                              lat + np.minimum(excess, need * rho) / rho, lat)
            T      = np.where(mask_m,
                              T_melt + np.maximum(0.0, excess - need * rho)
                              / (rho * cp_m + 1e-30), T)
        # decomp
        mask_d = (~abl) & (T >= T_abl) & (lat >= L_melt)
        if np.any(mask_d):
            cp_a  = cp_fn(T)
            exc_a = rho * cp_a * np.maximum(T - T_abl, 0.0)
            need_a= np.maximum(L_decomp - (lat - L_melt), 0.0)
            lat   = np.where(mask_d,
                             lat + np.minimum(exc_a, need_a * rho) / rho, lat)
            T     = np.where(mask_d,
                             T_abl + np.maximum(0.0, exc_a - need_a * rho)
                             / (rho * cp_a + 1e-30), T)
        abl = abl | ((~abl) & (lat >= L_melt + L_decomp))
        T   = np.where(abl, T_abl, T)

        # ── 3. Heat conduction: CN in z, explicit in r ───────────
        for step in range(N_step):
            in_pulse = step < N_on
            Q_now    = Q_sp * (1.0 if in_pulse else 0.0) * (~abl)

            # Explicit r-laplacian (evaluated at current T)
            lap_r  = _explicit_r_laplacian(T, r, dr, k_func)

            # Q_half = contribution to rhs: explicit lap_r + Q_source
            Q_rhs  = lap_r + Q_now          # (N,N)

            # CN-z half-step
            T_new  = _cn_z_step(T, dz, dt, rho, cp_fn, k_func, Q_rhs * dt)

            # Ablation guard
            T_new  = np.where(abl, T_abl, T_new)
            T_new  = np.clip(T_new, T_AMB - 1.0, None)
            T      = T_new

        # ── 4. Per-pulse metrics ──────────────────────────────────
        haz_mask = np.any(T >= T_melt, axis=1)
        haz_r    = float(r[haz_mask].max()) if haz_mask.any() else 0.0
        abl_z    = float(np.sum(np.any(abl, axis=0)) * dz)  # crude depth

        alpha_L  = 150e-6 if material == "PP" else 200e-6
        dT_surf  = np.maximum(T[:, 0] - T_AMB, 0.0)
        rim_h    = float(np.trapz(dT_surf * alpha_L, dx=dr) * 1e6)

        haz_history.append(haz_r * 1e6)
        depth_history.append(abl_z * 1e6)
        rim_history.append(rim_h)

        if pulse_n in snapshot_pulses:
            snapshots[pulse_n] = T.copy()
            if verbose:
                print(f"    pulse {pulse_n:3d}  HAZ={haz_r*1e6:.1f} µm  "
                      f"rim={rim_h:.3f} µm")

    return {
        "r":             r,
        "z":             z,
        "snapshots":     snapshots,
        "haz_history":   np.array(haz_history),
        "depth_history": np.array(depth_history),
        "rim_history":   np.array(rim_history),
        "pulse_range":   np.arange(1, N_pulses + 1),
    }


# ═══════════════════════════════════════════════════════════════════
#  Cross-code comparison
# ═══════════════════════════════════════════════════════════════════

def compare_with_ppFDM(
    material: str,
    k_func_star: Callable[[np.ndarray], np.ndarray],
    snapshot_pulses: list[int] | None = None,
    N: int = 60,
) -> dict:
    """
    Cross-code verification protocol:
      1. Load ppFDM T-snapshots (generated by export_training_data.py with k_lit)
      2. Run CN-FDM with k*(T) provided by inverse PINN
      3. Compute field-wise MARE between the two solvers

    If MARE < ~5%, the recovered k*(T) is independently validated.

    Returns:
      dict with keys:
        pulse_numbers, ppFDM_snaps, cn_snaps, mare_per_pulse, mean_mare
    """
    if snapshot_pulses is None:
        snapshot_pulses = [1, 5, 10, 20, 35, 50]

    # ── Load ppFDM reference ─────────────────────────────────────
    snap_path = FDM_OUT / f"{material}_snapshots.npz"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"{snap_path} not found.  Run export_training_data.py first.")
    ppfdm_data = np.load(snap_path)

    # ── Run CN-FDM with k*(T) ────────────────────────────────────
    print(f"\n  Running CN-FDM({material}) with k*(T) from inverse PINN ...")
    cn_result = run_cn_fdm(material, k_func=k_func_star,
                           N=N, snapshot_pulses=snapshot_pulses)

    # ── Compare T fields ─────────────────────────────────────────
    mare_per_pulse = []
    ppFDM_snaps    = {}
    cn_snaps       = {}

    for p in snapshot_pulses:
        key = f"T_pulse{p}"
        if key not in ppfdm_data:
            continue
        T_pp = ppfdm_data[key].astype(np.float64)   # (Nr_pp, Nz_pp)
        T_cn = cn_result["snapshots"].get(p)
        if T_cn is None:
            continue

        # Interpolate CN result to ppFDM grid if sizes differ
        if T_cn.shape != T_pp.shape:
            from scipy.interpolate import RegularGridInterpolator
            r_cn = cn_result["r"]
            z_cn = cn_result["z"]
            r_pp = ppfdm_data["r"]
            z_pp = ppfdm_data["z"]
            interp = RegularGridInterpolator(
                (r_cn, z_cn), T_cn, method="linear",
                bounds_error=False, fill_value=T_AMB)
            Rg_pp, Zg_pp = np.meshgrid(r_pp, z_pp, indexing="ij")
            pts  = np.column_stack([Rg_pp.ravel(), Zg_pp.ravel()])
            T_cn = interp(pts).reshape(T_pp.shape)

        mask = T_pp > T_AMB + 5.0
        if mask.sum() == 0:
            mare = 0.0
        else:
            mare = float(np.mean(
                np.abs(T_cn[mask] - T_pp[mask]) / (T_pp[mask] - T_AMB + 1.0)
            ) * 100.0)

        mare_per_pulse.append(mare)
        ppFDM_snaps[p] = T_pp
        cn_snaps[p]    = T_cn
        print(f"    pulse {p:3d}  MARE(CN(k*) vs ppFDM(k_lit)) = {mare:.2f}%")

    mean_mare = float(np.mean(mare_per_pulse)) if mare_per_pulse else float("nan")
    print(f"\n  Mean MARE over all snapshots: {mean_mare:.2f}%")

    return {
        "pulse_numbers":   snapshot_pulses,
        "ppFDM_snaps":     ppFDM_snaps,
        "cn_snaps":        cn_snaps,
        "mare_per_pulse":  np.array(mare_per_pulse),
        "mean_mare":       mean_mare,
        "r_cn":            cn_result["r"],
        "z_cn":            cn_result["z"],
    }


# ═══════════════════════════════════════════════════════════════════
#  KNet numpy wrapper (to call KNet from non-PyTorch code)
# ═══════════════════════════════════════════════════════════════════

def knet_numpy_fn(knet_state_dict: dict, T_min: float = 25.0,
                  T_max: float = 700.0):
    """
    Return a numpy-callable k_func from a saved KNet state dict.
    Avoids importing torch in pure-numpy scripts.

    Usage:
        results = np.load('fdm_output/inverse_kT_results.npz')
        # rebuild KNet from saved arrays (see inverse_pinn_kT.py for
        # how state is saved) — typically called from generate_si11_kT.py
    """
    import torch        # local import so cn_fdm_reference can run without torch
    import torch.nn as nn

    class _KNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.T_min = T_min
            self.T_max = T_max
            self.net = nn.Sequential(
                nn.Linear(1, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, 1),
            )

        def forward(self, T):
            T_n = (T - self.T_min) / (self.T_max - self.T_min + 1e-8)
            T_n = T_n.unsqueeze(-1) if T_n.dim() == 1 else T_n.reshape(-1, 1)
            raw = self.net(T_n).squeeze(-1)
            return (torch.nn.functional.softplus(raw) * 0.25 + 0.05).clamp(0.05, 0.80)

    net = _KNet()
    # Convert numpy arrays in state dict back to tensors
    state = {k: torch.tensor(v) for k, v in knet_state_dict.items()}
    net.load_state_dict(state)
    net.eval()

    def k_fn(T_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            T_t = torch.tensor(T_np.astype(np.float32))
            return net(T_t).numpy().astype(np.float64)

    return k_fn


# ═══════════════════════════════════════════════════════════════════
#  Quick self-test CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Cross-code CN-FDM verification (uses literature k(T))")
    parser.add_argument("--mat", choices=["PP", "PE", "both"], default="PP")
    parser.add_argument("--N",   type=int, default=60)
    parser.add_argument("--compare", action="store_true",
                        help="Run cross-code comparison using k*(T) from "
                             "fdm_output/inverse_kT_results.npz")
    parser.add_argument("--control", action="store_true",
                        help="[M2] Control experiment: CN(k_lit) vs ppFDM(k_lit) "
                             "— isolates solver-scheme discrepancy from k*(T) errors")
    args = parser.parse_args()

    mats = ["PP", "PE"] if args.mat == "both" else [args.mat]

    if args.control:
        # ── M2 Control: CN(k_lit) vs ppFDM(k_lit) ───────────────────
        print("\n" + "="*62)
        print("  M2 CONTROL EXPERIMENT: CN(k_lit)  vs  ppFDM(k_lit)")
        print("  Isolates solver-scheme discrepancy from k*(T) errors")
        print("="*62)
        for mat in mats:
            k_lit_fn = k_pp_lit if mat == "PP" else k_pe_lit
            cmp = compare_with_ppFDM(mat, k_func_star=k_lit_fn, N=args.N)
            mare_arr  = cmp["mare_per_pulse"]
            pulse_arr = np.array(cmp["pulse_numbers"][:len(mare_arr)], dtype=float)
            pfx = mat.lower()
            np.save(FDM_OUT / f"cn_{pfx}_control_mare.npy", mare_arr)
            np.save(FDM_OUT / f"cn_{pfx}_control_pulses.npy", pulse_arr)
            print(f"\n  {mat} Control: CN(k_lit) vs ppFDM(k_lit)")
            print(f"  Mean MARE = {cmp['mean_mare']:.2f}%")
            print(f"  Saved: cn_{pfx}_control_mare.npy")
        print("\n  Conclusion: If control MARE ≈ k* MARE, discrepancy is")
        print("  dominated by solver scheme, NOT by k*(T) errors.")

    elif args.compare:
        # ── Cross-code verification with k*(T) from inverse PINN ────
        from scipy.interpolate import interp1d

        npz_path = FDM_OUT / "inverse_kT_results.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} not found. Run inverse_pinn_kT.py first.")

        inv_data = np.load(npz_path)

        for mat in mats:
            pfx = mat.lower()
            T_knet  = inv_data[f"{pfx}_T"].astype(np.float64)
            k_knet  = inv_data[f"{pfx}_kstar"].astype(np.float64)
            k_interp = interp1d(T_knet, k_knet,
                                kind="linear", fill_value="extrapolate")

            def k_star_fn(T_np, _fi=k_interp):
                return np.clip(_fi(T_np.astype(np.float64)), 0.05, 0.80)

            print(f"\n{'='*62}")
            print(f"  Cross-code comparison: CN-FDM({mat}) with k*(T)")
            print(f"{'='*62}")

            cmp = compare_with_ppFDM(mat, k_func_star=k_star_fn, N=args.N)

            # Save MARE results for generate_si11_kT.py (panel d)
            mare_arr  = cmp["mare_per_pulse"]
            pulse_arr = np.array(cmp["pulse_numbers"][:len(mare_arr)],
                                  dtype=float)

            np.save(FDM_OUT / f"cn_{pfx}_mare_per_pulse.npy",  mare_arr)
            np.save(FDM_OUT / f"cn_{pfx}_pulse_range.npy",     pulse_arr)
            print(f"  Saved: cn_{pfx}_mare_per_pulse.npy  "
                  f"(mean MARE = {cmp['mean_mare']:.2f}%)")

            # Save pulse-20 T-field for panel (c)
            p20 = cmp["cn_snaps"].get(20)
            if p20 is not None:
                np.save(FDM_OUT / f"cn_{pfx}_pulse20.npy", p20)
                print(f"  Saved: cn_{pfx}_pulse20.npy")

    else:
        # ── Self-test with literature k(T) ───────────────────────────
        for mat in mats:
            k_lit_fn = k_pp_lit if mat == "PP" else k_pe_lit
            print(f"\nSelf-test: running CN-FDM({mat}) with literature k(T) ...")
            res = run_cn_fdm(mat, k_func=k_lit_fn, N=args.N,
                             snapshot_pulses=[1, 10, 50])
            print(f"\nHAZ @ pulse 50: {res['haz_history'][-1]:.1f} µm")
            print(f"Rim @ pulse 50: {res['rim_history'][-1]:.3f} µm")
            print(f"CN-FDM({mat}) self-test complete.")

