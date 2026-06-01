#!/usr/bin/env python3
"""wu_comparison.py
Run FDM simulation with Wu et al. (2022) laser parameters and compare
predicted kerf depth, kerf width, and HAZ with their experimental data.

Wu et al. 2022: "High-quality cutting of PP film by UV nanosecond laser
based on thermal ablation", Optics & Laser Technology, 147, 107600.
  - Laser: Nd:YAG 355 nm, 16 ± 2 ns, beam diameter D_s ≈ 11.8 µm (w₀ ≈ 5.9 µm)
  - PP film: 200 µm
  - Conditions: 25–125 kHz, 50–250 mm/s scanning speed

Approach:
  Fixed-point FDM approximation of scanning. Effective overlapping pulses:
    N_eff = D_beam × f_rep / v_scan
  Each condition uses Wu's actual repetition rate for inter-pulse cooling.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "fdm_output"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Wu et al. (2022) laser & geometry parameters
# ─────────────────────────────────────────────────────────────────
W0_WU      = 5.9e-6      # beam waist 1/e² radius [m]  (D_s ≈ 11.8 µm)
D_BEAM     = 11.8e-6     # beam diameter [m]
H_FILM_WU  = 200e-6      # PP film thickness [m]
R_MAX_WU   = 80e-6       # simulation domain radius [m]
T_AMB      = 25.0        # °C

# Convective & radiative BCs
H_CONV     = 25.0        # W/(m²·K)
EMISS      = 0.95
SB         = 5.67e-8     # W/(m²·K⁴)

# PP material (identical to main simulation)
RHO        = 910.0
T_MELT     = 165.0
T_ABL      = 350.0
ALP_ABS    = 0.6e5       # m⁻¹ absorption coefficient
ALPL       = 150e-6      # 1/K CTE
L_MELT     = 100e3       # J/kg
L_DECOMP   = 300e3       # J/kg

# ─────────────────────────────────────────────────────────────────
# Wu Table 1: pulse energy vs repetition rate (interpolated for 25 kHz)
# ─────────────────────────────────────────────────────────────────
WU_EP = {25: 1.39, 50: 2.20, 75: 2.23, 100: 1.77, 125: 1.12}  # µJ

WU_SPEEDS = [50, 100, 150, 200, 250]   # mm/s

# Wu Table 3: kerf width Ws (µm),  rows = rep_rate, cols = speed
WU_KERF = {
    25:  [46.17, 32.0,  31.83, 21.67, 17.50],
    50:  [51.67, 42.4,  39.0,  36.2,  27.5],
    75:  [68.99, 44.17, 42.67, 39.31, 29.17],
    100: [63.06, 41.17, 40.33, 35.67, 19.64],
    125: [60.2,  30.0,  24.17, 19.8,  18.33],
}

# Wu Table 4: HAZ width Wh (µm)
WU_HAZ = {
    25:  [30.38, 26.46, 28.42, 23.52, 21.56],
    50:  [28.42, 38.22, 35.28, 31.36, 26.5],
    75:  [46.1,  31.36, 26.46, 21.56, 18.66],
    100: [42.98, 25.4,  21.5,  18.62, 12.68],
    125: [43.12, 23.54, 19.6,  17.74, 18.62],
}

# Wu Section 3.2: kerf depth at optimal (100 kHz, 250 mm/s, repeat=1)
WU_DEPTH_OPTIMAL = 26.4   # µm


# ─────────────────────────────────────────────────────────────────
# Material property functions (match export_training_data.py exactly)
# ─────────────────────────────────────────────────────────────────
def cp_pp(T):
    cb = 1780.0 + 570.0 / (1.0 + np.exp(-(T - 140.0) / 50.0))
    return cb + 250.0 * np.exp(-0.5 * ((T - 165.0) / 8.5) ** 2)

def k_pp(T):
    kc = np.minimum(0.22 * 298.15 / np.maximum(T + 273.15, 200.0), 0.24)
    km = np.maximum(0.155 - 3e-5 * np.maximum(T - T_MELT, 0.0), 0.13)
    fm = 1.0 / (1.0 + np.exp(-(T - T_MELT) / 10.0))
    return kc * (1.0 - fm) + km * fm


# ─────────────────────────────────────────────────────────────────
# FDM solver with Wu parameters
# ─────────────────────────────────────────────────────────────────
def run_fdm_wu(Ep_J, f_rep_Hz, N_pulses, N=160):
    """Explicit FDM with Wu's beam/geometry parameters."""
    r  = np.linspace(0.0, R_MAX_WU, N)
    z  = np.linspace(0.0, H_FILM_WU, N)
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    f_peak = 2.0 * Ep_J / (np.pi * W0_WU ** 2)
    f_r    = f_peak * np.exp(-2.0 * r ** 2 / W0_WU ** 2)

    T   = np.full((N, N), T_AMB, dtype=np.float64)
    abl = np.zeros((N, N), dtype=bool)
    lat = np.zeros((N, N), dtype=np.float64)

    alpha0  = k_pp(T_AMB) / (RHO * cp_pp(T_AMB))
    dt_cfl  = 0.3 / (alpha0 * (1.0 / dr ** 2 + 1.0 / dz ** 2))
    pulse_dt = 1.0 / f_rep_Hz
    n_sub   = max(int(np.ceil(pulse_dt / dt_cfl)) + 4, 20)
    dt      = pulse_dt / n_sub

    depth_hist = []
    haz_hist   = []

    for p in range(N_pulses):
        # ── Heat deposition (Beer-Lambert) ──
        any_solid = np.any(~abl, axis=1)
        front_j   = np.where(any_solid, np.argmax(~abl, axis=1), N).astype(np.int32)
        z_front   = np.where(front_j < N, z[np.minimum(front_j, N - 1)], H_FILM_WU)

        z_rel = np.maximum(z[np.newaxis, :] - z_front[:, np.newaxis], 0.0)
        Q_vol = f_r[:, np.newaxis] * ALP_ABS * np.exp(-ALP_ABS * z_rel) * (~abl)
        T = T + Q_vol / (RHO * cp_pp(T))

        # ── Latent heat: melting ──
        mask_m = (~abl) & (T > T_MELT) & (lat < L_MELT)
        if np.any(mask_m):
            cp_m   = cp_pp(T)
            excess = RHO * cp_m * (T - T_MELT)
            need   = L_MELT - lat
            cons   = np.minimum(excess, need * RHO)
            lat = np.where(mask_m, lat + cons / RHO, lat)
            T   = np.where(mask_m,
                           T_MELT + np.maximum(0.0, excess - need * RHO) / (RHO * cp_m), T)

        # ── Latent heat: decomposition + ablation ──
        mask_d = (~abl) & (T >= T_ABL) & (lat >= L_MELT)
        if np.any(mask_d):
            cp_a   = cp_pp(T)
            exc_a  = RHO * cp_a * np.maximum(T - T_ABL, 0.0)
            need_a = np.maximum(L_DECOMP - (lat - L_MELT), 0.0)
            cons_a = np.where(need_a > 0, np.minimum(exc_a, need_a * RHO), 0.0)
            lat = np.where(mask_d, lat + cons_a / RHO, lat)
            T   = np.where(mask_d,
                           T_ABL + np.maximum(0.0, exc_a - need_a * RHO) / (RHO * cp_a), T)
        new_abl = (~abl) & (lat >= L_MELT + L_DECOMP)
        abl = abl | new_abl
        T   = np.where(abl, T_ABL, T)

        # ── Heat conduction (explicit FDM) ──
        for _ in range(n_sub):
            kT       = k_pp(T)
            alpha_2d = kT / (RHO * cp_pp(T))
            T_new    = T.copy()

            # interior
            Ti_p = np.where(abl[2:,    :],    T[1:-1, :],    T[2:,    :])
            Ti_m = np.where(abl[:-2,   :],    T[1:-1, :],    T[:-2,   :])
            Tj_p = np.where(abl[1:-1, 2:],    T[1:-1, 1:-1], T[1:-1, 2:])
            Tj_m = np.where(abl[1:-1, :-2],   T[1:-1, 1:-1], T[1:-1, :-2])

            r_int  = r[1:-1, np.newaxis]
            d2Tdr2 = (Ti_p[:, 1:-1] - 2.0 * T[1:-1, 1:-1] + Ti_m[:, 1:-1]) / dr ** 2
            dTdr   = (Ti_p[:, 1:-1] - Ti_m[:, 1:-1]) / (2.0 * dr)
            d2Tdz2 = (Tj_p - 2.0 * T[1:-1, 1:-1] + Tj_m) / dz ** 2
            lap    = d2Tdr2 + dTdr / r_int + d2Tdz2
            T_new[1:-1, 1:-1] = np.where(
                abl[1:-1, 1:-1], T[1:-1, 1:-1],
                T[1:-1, 1:-1] + alpha_2d[1:-1, 1:-1] * dt * lap)

            # r = 0 axis (L'Hôpital)
            T1j   = np.where(abl[1, 1:-1],  T[0, 1:-1], T[1, 1:-1])
            Tj_p0 = np.where(abl[0, 2:],    T[0, 1:-1], T[0, 2:])
            Tj_m0 = np.where(abl[0, :-2],   T[0, 1:-1], T[0, :-2])
            d2r0  = (T1j - T[0, 1:-1]) / dr ** 2
            d2z0  = (Tj_p0 - 2.0 * T[0, 1:-1] + Tj_m0) / dz ** 2
            T_new[0, 1:-1] = np.where(
                abl[0, 1:-1], T[0, 1:-1],
                T[0, 1:-1] + alpha_2d[0, 1:-1] * dt * (2.0 * d2r0 + d2z0))

            # BCs
            T_new[-1, :] = T_new[-2, :]                        # r = R_max adiabatic
            kz0 = k_pp(T_new[:, 0])
            TS  = T_new[:, 0]; TsK = TS + 273.15; TaK = T_AMB + 273.15
            q_bc = H_CONV * (TS - T_AMB) + EMISS * SB * (TsK ** 4 - TaK ** 4)
            T_new[:, 0] = np.where(abl[:, 0], T[:, 0],
                                   T_new[:, 1] - q_bc * dz / kz0)   # z = 0 surface
            T_new[:, -1] = T_new[:, -2]                        # z = H_FILM bottom
            T = T_new

        # ── Per-pulse metrics ──
        depth_hist.append(float(np.sum(abl[0, :]) * dz * 1e6))
        haz_r = 0.0
        for ii in range(N - 1, -1, -1):
            if (not np.any(abl[ii, :])) and T[ii, :].max() > T_MELT:
                haz_r = r[ii] * 1e6
                break
        haz_hist.append(haz_r)

    # ablation diameter
    abl_r_idx = np.where(np.any(abl, axis=1))[0]
    abl_dia   = float(2 * r[abl_r_idx[-1]] * 1e6) if len(abl_r_idx) > 0 else 0.0

    # rim height (thermal expansion based)
    rim_max = 0.0
    for ii in range(N):
        if not np.any(abl[ii, :]):
            h_i = float(ALPL * np.sum((T[ii, :] - T_AMB) * dz) * 1e6)
            rim_max = max(rim_max, h_i)

    return {
        'depth_um':    depth_hist[-1] if depth_hist else 0.0,
        'abl_dia_um':  abl_dia,
        'haz_radius_um': haz_hist[-1] if haz_hist else 0.0,
        'rim_um':      rim_max,
        'depth_hist':  np.array(depth_hist),
        'haz_hist':    np.array(haz_hist),
    }


# ─────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────
def main():
    REP_RATES = [25, 50, 75, 100, 125]   # kHz
    SPEEDS    = [50, 100, 150, 200, 250]  # mm/s
    N_GRID    = 160

    print("=" * 80)
    print("FDM simulation with Wu et al. (2022) parameters")
    print(f"  Beam: w₀ = {W0_WU*1e6:.1f} µm,  PP film = {H_FILM_WU*1e6:.0f} µm")
    print(f"  Grid: N = {N_GRID},  R_max = {R_MAX_WU*1e6:.0f} µm")
    print("=" * 80)

    rows = []
    t_total = time.time()

    for f_kHz in REP_RATES:
        Ep_uJ = WU_EP[f_kHz]
        Ep_J  = Ep_uJ * 1e-6
        f_Hz  = f_kHz * 1e3
        fluence = 2 * Ep_J / (np.pi * W0_WU ** 2)  # J/m²

        for j, v_mms in enumerate(SPEEDS):
            v_ms   = v_mms * 1e-3
            N_eff  = D_BEAM * f_Hz / v_ms
            N_eff_int = max(1, round(N_eff))

            print(f"\n  [{f_kHz} kHz, {v_mms} mm/s]  Ep={Ep_uJ:.2f} µJ  "
                  f"F₀={fluence/1e4:.2f} J/cm²  N_eff={N_eff:.1f}→{N_eff_int}", end="  ")

            t0 = time.time()
            res = run_fdm_wu(Ep_J, f_Hz, N_eff_int, N=N_GRID)
            dt_ = time.time() - t0

            # Wu experimental data
            wu_kerf = WU_KERF[f_kHz][j]
            wu_haz  = WU_HAZ[f_kHz][j]

            # Our: ablation diameter ≈ kerf width (Ws)
            # Our: HAZ_total = 2 × (haz_radius - ablation_radius)
            abl_radius = res['abl_dia_um'] / 2.0
            haz_total  = max(0.0, 2.0 * (res['haz_radius_um'] - abl_radius))

            print(f"depth={res['depth_um']:.1f} µm  "
                  f"Ws_pred={res['abl_dia_um']:.1f} vs {wu_kerf:.1f} µm  "
                  f"Wh_pred={haz_total:.1f} vs {wu_haz:.1f} µm  [{dt_:.1f}s]")

            rows.append({
                'f_kHz':       f_kHz,
                'v_mms':       v_mms,
                'Ep_uJ':       Ep_uJ,
                'F0_Jcm2':     fluence / 1e4,
                'N_eff':       N_eff_int,
                'depth_pred':  res['depth_um'],
                'Ws_pred':     res['abl_dia_um'],
                'Wh_pred':     haz_total,
                'rim_pred':    res['rim_um'],
                'Ws_wu':       wu_kerf,
                'Wh_wu':       wu_haz,
                'Ws_err_pct':  (res['abl_dia_um'] - wu_kerf) / wu_kerf * 100 if wu_kerf > 0 else 0,
                'Wh_err_pct':  (haz_total - wu_haz) / wu_haz * 100 if wu_haz > 0 else 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'wu_comparison.csv', index=False, float_format='%.2f')
    print(f"\n  Saved wu_comparison.csv  ({time.time()-t_total:.0f}s total)")

    # ── Summary statistics ──
    print("\n" + "=" * 80)
    print("SUMMARY: FDM predictions vs Wu et al. (2022) experimental data")
    print("=" * 80)

    # Kerf width comparison
    ws_mae = df['Ws_err_pct'].abs().mean()
    wh_mae = df['Wh_err_pct'].abs().mean()
    print(f"\n  Kerf width (Ws):  mean |error| = {ws_mae:.1f}%")
    print(f"  HAZ width  (Wh):  mean |error| = {wh_mae:.1f}%")

    # Optimal condition depth
    opt = df[(df['f_kHz'] == 100) & (df['v_mms'] == 250)]
    if not opt.empty:
        d_pred = opt['depth_pred'].values[0]
        d_err  = (d_pred - WU_DEPTH_OPTIMAL) / WU_DEPTH_OPTIMAL * 100
        print(f"\n  Kerf depth (optimal 100 kHz, 250 mm/s):")
        print(f"    Predicted = {d_pred:.1f} µm  vs  Wu = {WU_DEPTH_OPTIMAL} µm  "
              f"(error = {d_err:+.1f}%)")

    # ── Formatted comparison table ──
    print("\n" + "─" * 95)
    print(f"{'f(kHz)':>7} {'v(mm/s)':>8} {'Ep(µJ)':>7} {'N_eff':>5} "
          f"{'Ws_pred':>8} {'Ws_wu':>7} {'Ws_err%':>8} "
          f"{'Wh_pred':>8} {'Wh_wu':>7} {'Wh_err%':>8} {'depth':>7}")
    print("─" * 95)
    for _, r in df.iterrows():
        print(f"{r['f_kHz']:7.0f} {r['v_mms']:8.0f} {r['Ep_uJ']:7.2f} {r['N_eff']:5.0f} "
              f"{r['Ws_pred']:8.1f} {r['Ws_wu']:7.1f} {r['Ws_err_pct']:+8.1f} "
              f"{r['Wh_pred']:8.1f} {r['Wh_wu']:7.1f} {r['Wh_err_pct']:+8.1f} "
              f"{r['depth_pred']:7.1f}")
    print("─" * 95)

    # ── Also run original parameters for comparison ──
    print("\n\n" + "=" * 80)
    print("REFERENCE: Original simulation (This work)")
    print(f"  w₀={35} µm, Ep=100 µJ, f=10 kHz, H_film=100 µm, N_pulses=20")
    print("=" * 80)

    # Run original parameters with original solver dimensions
    # (re-import from export_training_data would be cleaner, but for standalone:)
    print("  (See export_training_data.py output for original baseline values)")


if __name__ == '__main__':
    main()
