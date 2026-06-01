#!/usr/bin/env python3
"""export_training_data.py
Vectorized numpy FDM that generates PINN training data.

Outputs (in fdm_output/):
  PP_snapshots.npz  -- T fields at [1,5,10,20,35,50] pulses for PP
  PE_snapshots.npz  -- same for PE
  PP_multiEp.csv    -- final ablation depth vs Ep (30..300 µJ) for PP
  PE_multiEp.csv    -- same for PE

Run time est.: ~3–8 min (vectorized numpy, N=80 grid).
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "fdm_output"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Process constants  (match ppFDM exactly)
# ─────────────────────────────────────────────────────────────────
R_MAX   = 200e-6   # m  domain radius
H_FILM  = 100e-6   # m  film thickness
W0      = 35e-6    # m  1/e² beam radius
F_REP   = 10e3     # Hz repetition rate
T_AMB   = 25.0     # °C ambient
H_CONV  = 25.0     # W/(m²·K) convective coefficient
EMISS   = 0.95
SB      = 5.67e-8  # W/(m²·K⁴)

# PP
RHO_PP   = 910.0
T_M_PP   = 165.0
T_ABL_PP = 350.0
ALP_PP   = 0.6e5   # m⁻¹ absorption coefficient (fixed, matches ppFDM)
ALPL_PP  = 150e-6  # 1/K CTE
L_M_PP   = 100e3   # J/kg latent melt
L_D_PP   = 300e3   # J/kg latent decomp
NU_PP    = 0.42

# PE (HDPE)
RHO_PE   = 950.0
T_M_PE   = 132.0
T_ABL_PE = 350.0
ALP_PE   = 0.4e5
ALPL_PE  = 200e-6
L_M_PE   = 200e3
L_D_PE   = 280e3
NU_PE    = 0.44


# ─────────────────────────────────────────────────────────────────
# Temperature-dependent material properties  (match ppFDM)
# ─────────────────────────────────────────────────────────────────
def cp_pp(T):
    cb = 1780.0 + 570.0 / (1.0 + np.exp(-(T - 140.0) / 50.0))
    return cb + 250.0 * np.exp(-0.5 * ((T - 165.0) / 8.5) ** 2)

def k_pp(T):
    kc = np.minimum(0.22 * 298.15 / np.maximum(T + 273.15, 200.0), 0.24)
    km = np.maximum(0.155 - 3e-5 * np.maximum(T - T_M_PP, 0.0), 0.13)
    fm = 1.0 / (1.0 + np.exp(-(T - T_M_PP) / 10.0))
    return kc * (1.0 - fm) + km * fm

def cp_pe(T):
    cb = 1900.0 + 400.0 / (1.0 + np.exp(-(T - 110.0) / 40.0))
    return cb + 370.0 * np.exp(-0.5 * ((T - 132.0) / 7.0) ** 2)

def k_pe(T):
    kc = np.minimum(0.45 * 298.15 / np.maximum(T + 273.15, 200.0), 0.50)
    km = np.maximum(0.35 - 2e-5 * np.maximum(T - T_M_PE, 0.0), 0.30)
    fm = 1.0 / (1.0 + np.exp(-(T - T_M_PE) / 8.0))
    return kc * (1.0 - fm) + km * fm


# ─────────────────────────────────────────────────────────────────
# Vectorized FDM solver
# ─────────────────────────────────────────────────────────────────
def run_fdm(material: str, Ep: float, N: int = 80,
            N_pulses: int = 50,
            snapshot_pulses: list | None = None) -> dict:
    """
    Explicit FDM on uniform N×N grid.  Vectorized heat-conduction kernel.
    Returns dict of snapshots, per-pulse metrics, and final field.
    """
    if snapshot_pulses is None:
        snapshot_pulses = [1, 5, 10, 20, 35, 50]

    # material params
    if material == 'PP':
        rho, T_abl, T_melt = RHO_PP, T_ABL_PP, T_M_PP
        alpha_abs, alpha_L  = ALP_PP, ALPL_PP
        L_melt, L_decomp    = L_M_PP, L_D_PP
        cp_fn, k_fn         = cp_pp, k_pp
    else:
        rho, T_abl, T_melt = RHO_PE, T_ABL_PE, T_M_PE
        alpha_abs, alpha_L  = ALP_PE, ALPL_PE
        L_melt, L_decomp    = L_M_PE, L_D_PE
        cp_fn, k_fn         = cp_pe, k_pe

    r  = np.linspace(0.0, R_MAX, N)
    z  = np.linspace(0.0, H_FILM, N)
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    # pre-compute radial profile of surface fluence
    f_peak = 2.0 * Ep / (np.pi * W0 ** 2)   # J/m²
    f_r    = f_peak * np.exp(-2.0 * r ** 2 / W0 ** 2)  # (N,)

    # state
    T   = np.full((N, N), T_AMB, dtype=np.float64)
    abl = np.zeros((N, N), dtype=bool)
    lat = np.zeros((N, N), dtype=np.float64)

    # CFL substep count
    alpha0  = k_fn(T_AMB) / (rho * cp_fn(T_AMB))
    dt_cfl  = 0.3 / (alpha0 * (1.0 / dr ** 2 + 1.0 / dz ** 2))
    pulse_dt = 1.0 / F_REP          # 100 µs inter-pulse window
    n_sub   = max(int(np.ceil(pulse_dt / dt_cfl)) + 4, 20)
    dt      = pulse_dt / n_sub

    snapshots  = {}
    depth_hist = []
    haz_hist   = []
    rim_hist   = []

    for p in range(N_pulses):
        # ── 1. Instantaneous heat deposition (Beer-Lambert) ─────────────
        # ablation front per radial strip
        any_solid = np.any(~abl, axis=1)              # (N,) bool
        front_j   = np.where(any_solid,
                              np.argmax(~abl, axis=1),
                              N).astype(np.int32)      # (N,)
        z_front   = np.where(front_j < N, z[np.minimum(front_j, N-1)], H_FILM)  # (N,)

        # z_rel[i, j] = z[j] - z_front[i]  ≥ 0 for non-ablated cells
        Z_2d   = z[np.newaxis, :]               # (1, N)
        zf_2d  = z_front[:, np.newaxis]         # (N, 1)
        z_rel  = np.maximum(Z_2d - zf_2d, 0.0) # (N, N)

        Q_vol = (f_r[:, np.newaxis] * alpha_abs
                 * np.exp(-alpha_abs * z_rel) * (~abl))   # J/m³
        dT_src = Q_vol / (rho * cp_fn(T))
        T      = T + dT_src

        # ── 2. Latent heat buffering + ablation marking ─────────────────
        # melt latent
        mask_m = (~abl) & (T > T_melt) & (lat < L_melt)
        if np.any(mask_m):
            cp_m   = cp_fn(T)
            excess = rho * cp_m * (T - T_melt)
            need   = L_melt - lat
            cons   = np.minimum(excess, need * rho)
            lat    = np.where(mask_m, lat + cons / rho, lat)
            T      = np.where(mask_m,
                              T_melt + np.maximum(0.0, excess - need * rho) / (rho * cp_m),
                              T)

        # decomposition latent + ablation
        mask_d = (~abl) & (T >= T_abl) & (lat >= L_melt)
        if np.any(mask_d):
            cp_a    = cp_fn(T)
            exc_a   = rho * cp_a * np.maximum(T - T_abl, 0.0)
            need_a  = np.maximum(L_decomp - (lat - L_melt), 0.0)
            cons_a  = np.where(need_a > 0, np.minimum(exc_a, need_a * rho), 0.0)
            lat     = np.where(mask_d, lat + cons_a / rho, lat)
            T       = np.where(mask_d,
                               T_abl + np.maximum(0.0, exc_a - need_a * rho) / (rho * cp_a),
                               T)
        new_abl = (~abl) & (lat >= L_melt + L_decomp)
        abl     = abl | new_abl
        T       = np.where(abl, T_abl, T)

        # ── 3. Heat conduction  (vectorized explicit FDM) ───────────────
        for _ in range(n_sub):
            kT       = k_fn(T)
            alpha_2d = kT / (rho * cp_fn(T))   # (N, N)
            T_new    = T.copy()

            # --- interior r=1..N-2, z=1..N-2 ---
            # adiabatic reflection: ablated neighbor → use self T
            Ti_p = np.where(abl[2:,    :],    T[1:-1, :],    T[2:,    :])  # (N-2, N)
            Ti_m = np.where(abl[:-2,   :],    T[1:-1, :],    T[:-2,   :])  # (N-2, N)
            Tj_p = np.where(abl[1:-1, 2:],    T[1:-1, 1:-1], T[1:-1, 2:])  # (N-2, N-2)
            Tj_m = np.where(abl[1:-1, :-2],   T[1:-1, 1:-1], T[1:-1, :-2]) # (N-2, N-2)

            r_int = r[1:-1, np.newaxis]   # (N-2, 1)  – broadcast OK

            d2Tdr2 = (Ti_p[:, 1:-1] - 2.0 * T[1:-1, 1:-1] + Ti_m[:, 1:-1]) / dr ** 2
            dTdr   = (Ti_p[:, 1:-1] - Ti_m[:, 1:-1]) / (2.0 * dr)
            d2Tdz2 = (Tj_p - 2.0 * T[1:-1, 1:-1] + Tj_m) / dz ** 2

            lap    = d2Tdr2 + dTdr / r_int + d2Tdz2
            T_new[1:-1, 1:-1] = np.where(
                abl[1:-1, 1:-1],
                T[1:-1, 1:-1],
                T[1:-1, 1:-1] + alpha_2d[1:-1, 1:-1] * dt * lap
            )

            # --- r = 0 axis (L'Hôpital: 1/r · dT/dr → d²T/dr²) ---
            T1j    = np.where(abl[1, 1:-1],  T[0, 1:-1], T[1, 1:-1])
            Tj_p0  = np.where(abl[0, 2:],    T[0, 1:-1], T[0, 2:])
            Tj_m0  = np.where(abl[0, :-2],   T[0, 1:-1], T[0, :-2])
            d2r0   = (T1j - T[0, 1:-1]) / dr ** 2
            d2z0   = (Tj_p0 - 2.0 * T[0, 1:-1] + Tj_m0) / dz ** 2
            T_new[0, 1:-1] = np.where(
                abl[0, 1:-1],
                T[0, 1:-1],
                T[0, 1:-1] + alpha_2d[0, 1:-1] * dt * (2.0 * d2r0 + d2z0)
            )

            # --- r = R_max: adiabatic Neumann ---
            T_new[-1, :] = T_new[-2, :]

            # --- z = 0 surface BC: Newton cooling + Stefan-Boltzmann ---
            kz0 = k_fn(T_new[:, 0])
            TS  = T_new[:, 0]
            TsK = TS + 273.15
            TaK = T_AMB + 273.15
            q_bc = H_CONV * (TS - T_AMB) + EMISS * SB * (TsK ** 4 - TaK ** 4)
            T_new[:, 0] = np.where(
                abl[:, 0], T[:, 0],
                T_new[:, 1] - q_bc * dz / kz0
            )

            # --- z = H bottom: adiabatic ---
            T_new[:, -1] = T_new[:, -2]

            T = T_new

        # ── 4. Per-pulse metrics ─────────────────────────────────────────
        depth0 = float(np.sum(abl[0, :]) * dz * 1e6)   # µm at r=0
        depth_hist.append(depth0)

        haz_r = 0.0
        for ii in range(N - 1, -1, -1):
            if (not np.any(abl[ii, :])) and T[ii, :].max() > T_melt:
                haz_r = r[ii] * 1e6
                break
        haz_hist.append(haz_r)

        rim_max = 0.0
        for ii in range(N):
            if not np.any(abl[ii, :]):
                h_i = float(alpha_L * np.sum((T[ii, :] - T_AMB) * dz) * 1e6)
                rim_max = max(rim_max, h_i)
        rim_hist.append(rim_max)

        if (p + 1) in snapshot_pulses:
            snapshots[p + 1] = T.copy()

    # ── 5. Final rim profile ─────────────────────────────────────────────
    rim_profile = np.zeros(N)
    for ii in range(N):
        if np.any(abl[ii, :]):
            rim_profile[ii] = -float(np.sum(abl[ii, :]) * dz * 1e6)
        else:
            rim_profile[ii] = float(alpha_L * np.sum((T[ii, :] - T_AMB) * dz) * 1e6)

    haz_final = 0.0
    for ii in range(N - 1, -1, -1):
        if not np.any(abl[ii, :]) and T[ii, :].max() > T_melt:
            haz_final = r[ii] * 1e6
            break

    return {
        'T_grid':      T,
        'abl_grid':    abl.astype(np.uint8),
        'rim_profile': rim_profile,
        'r': r, 'z': z,
        'snapshots':   snapshots,
        'depth_hist':  np.array(depth_hist),
        'haz_hist':    np.array(haz_hist),
        'rim_hist':    np.array(rim_hist),
        'haz_final':   haz_final,
        'pulses':      np.arange(1, N_pulses + 1),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    N_GRID = 80      # grid resolution for training data
    EP_NOM = 100e-6  # nominal pulse energy

    # ── PP snapshots ──────────────────────────────────────────────
    print("=== PP: generating FDM snapshots (N=%d, 50 pulses) ===" % N_GRID)
    t0 = time.time()
    res_pp = run_fdm('PP', EP_NOM, N=N_GRID, snapshot_pulses=[1, 5, 10, 20, 35, 50])
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"depth={res_pp['depth_hist'][-1]:.1f}µm  "
          f"HAZ={res_pp['haz_final']:.1f}µm  "
          f"rim={res_pp['rim_hist'][-1]:.3f}µm")

    snap_kw = {'r': res_pp['r'], 'z': res_pp['z'],
               'pulses': res_pp['pulses'],
               'depth_hist': res_pp['depth_hist'],
               'haz_hist':   res_pp['haz_hist'],
               'rim_hist':   res_pp['rim_hist'],
               'T_grid':     res_pp['T_grid'],
               'abl_grid':   res_pp['abl_grid'],
               'rim_profile':res_pp['rim_profile']}
    for p, T_snap in res_pp['snapshots'].items():
        snap_kw[f'T_pulse{p}'] = T_snap
    np.savez_compressed(OUT / 'PP_snapshots.npz', **snap_kw)
    print(f"  Saved PP_snapshots.npz")

    # ── PE snapshots ──────────────────────────────────────────────
    print("=== PE: generating FDM snapshots (N=%d, 50 pulses) ===" % N_GRID)
    t0 = time.time()
    res_pe = run_fdm('PE', EP_NOM, N=N_GRID, snapshot_pulses=[1, 5, 10, 20, 35, 50])
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"depth={res_pe['depth_hist'][-1]:.1f}µm  "
          f"HAZ={res_pe['haz_final']:.1f}µm  "
          f"rim={res_pe['rim_hist'][-1]:.3f}µm")

    snap_kw_pe = {'r': res_pe['r'], 'z': res_pe['z'],
                  'pulses': res_pe['pulses'],
                  'depth_hist': res_pe['depth_hist'],
                  'haz_hist':   res_pe['haz_hist'],
                  'rim_hist':   res_pe['rim_hist'],
                  'T_grid':     res_pe['T_grid'],
                  'abl_grid':   res_pe['abl_grid'],
                  'rim_profile':res_pe['rim_profile']}
    for p, T_snap in res_pe['snapshots'].items():
        snap_kw_pe[f'T_pulse{p}'] = T_snap
    np.savez_compressed(OUT / 'PE_snapshots.npz', **snap_kw_pe)
    print(f"  Saved PE_snapshots.npz")

    # ── Multi-Ep ablation study (PP + PE, Ep = 30..300 µJ) ────────
    # Multi-Ep: run only 5 pulses to capture early ablation regime (depth varies with Ep)
    EP_vals = [20, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300]   # µJ

    for mat in ['PP', 'PE']:
        print(f"\n=== {mat}: multi-Ep sweep (N_pulses=5, {EP_vals[0]}..{EP_vals[-1]} µJ) ===")
        rows = []
        for ep_uj in EP_vals:
            Ep = ep_uj * 1e-6
            t0 = time.time()
            # N_pulses=5: captures pre-full-ablation regime where depth varies with Ep
            # Use fine radial grid (N=120, R_MAX implicit) to resolve narrow craters
            r_ep = run_fdm(mat, Ep, N=80, N_pulses=5, snapshot_pulses=[5])
            depth = float(r_ep['depth_hist'][-1])        # depth at r=0 after 5 pulses
            haz   = float(r_ep['haz_hist'][-1])
            rim   = float(r_ep['rim_hist'][-1])
            # ablation diameter = 2 * max r where any ablated cell exists
            abl_g = r_ep['abl_grid'].astype(bool)
            abl_r_idx = np.where(np.any(abl_g, axis=1))[0]
            abl_dia = float(2 * r_ep['r'][abl_r_idx[-1]] * 1e6) if len(abl_r_idx) > 0 else 0.0
            dt_   = time.time() - t0
            rows.append({'Ep_uJ': ep_uj, 'depth_um': depth, 'abl_dia_um': abl_dia,
                         'haz_um': haz, 'rim_um': rim})
            print(f"  Ep={ep_uj:4d} µJ  depth={depth:5.1f}µm  dia={abl_dia:5.1f}µm  "
                  f"HAZ={haz:5.1f}µm  rim={rim:.3f}µm  [{dt_:.1f}s]")
        pd.DataFrame(rows).to_csv(OUT / f'{mat}_multiEp.csv', index=False)
        print(f"  Saved {mat}_multiEp.csv")

    print("\n=== All FDM training data generated. ===")
    print(f"  Output directory: {OUT}")


if __name__ == '__main__':
    main()
