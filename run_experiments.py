#!/usr/bin/env python3
"""run_experiments.py — Complete M10 sweep + m1/m8 multi-seed experiments.

Multi-seed (m1/m8):
  Uses train_fast.py's train_temp_pinn() directly for each seed.
  Matches production pipeline: TempNet 6×64, data-only, 3000 epochs.

Loss-weight sweep (M10):
  Uses train_pinn.py's train_temporal_pinn() directly for each λ config
  at REDUCED 3000 epochs (vs 6000 full). This uses the exact same
  infrastructure: TempPINN 8×64, all-point data, PDE+BC from epoch 1.
  NOTE: This is slow (~8 min/config × 9 = ~72 min).
  Use --sweep-only or --seed-only flags to run selectively.
"""
from __future__ import annotations
import sys, io, time, json, argparse
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


def run_multiseed():
    """5-seed reproducibility using train_fast.py's exact pipeline."""
    from train_fast import train_temp_pinn, mare_np

    print("=" * 60)
    print("  m1/m8: MULTI-SEED (train_fast pipeline, 3000 ep)")
    print("=" * 60)

    SEEDS = [42, 123, 256, 512, 1024]
    results = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        r = train_temp_pinn('PP', n_epochs=3000)
        results.append({
            'seed': seed,
            'mare_p50': round(r['mare_50'], 4),
            'mare_p35': round(r['mare_35'], 4),
        })
        print(f"  >> seed={seed}  MARE_p50={r['mare_50']:.4f}%  "
              f"MARE_p35={r['mare_35']:.4f}%\n")

    mares = [x['mare_p50'] for x in results]
    mu, sigma = float(np.mean(mares)), float(np.std(mares))
    print(f"  Mean MARE_p50 = {mu:.4f}% ± σ = {sigma:.4f}%")
    return results, mu, sigma


def run_sweep():
    """Cardinal-neighbor sweep using train_pinn.py at full 6000 epochs.

    Runs 4 configs around the selected (0.03, 0.005) baseline:
      - vary lambda_PDE: {0.01, 0.10} at lambda_BC=0.005
      - vary lambda_BC:  {0.001, 0.02} at lambda_PDE=0.03
    The center (0.03, 0.005) is already stored in pinn_results.npz.
    """
    from train_pinn import train_temporal_pinn

    print("\n" + "=" * 60)
    print("  M10: CARDINAL SWEEP (train_pinn, 6000 ep)")
    print("=" * 60)

    CONFIGS = [
        (0.01, 0.005),   # reduce PDE weight
        (0.10, 0.005),   # increase PDE weight
        (0.03, 0.001),   # reduce BC weight
        (0.03, 0.02),    # increase BC weight
    ]
    results = []

    for lp, lb in CONFIGS:
        torch.manual_seed(42)
        np.random.seed(42)

        r = train_temporal_pinn(
            mat='PP', n_epochs=6000, lr=1.5e-3,
            lam_pde=lp, lam_bc=lb)

        results.append({
            'lam_pde': lp, 'lam_bc': lb,
            'mare_p50': round(r['mare_50'], 2),
            'mare_p35': round(r['mare_35'], 2),
        })
        print(f"  >> lam_PDE={lp:.2f} lam_BC={lb:.3f}  "
              f"MARE_p50={r['mare_50']:.2f}%  "
              f"MARE_p35={r['mare_35']:.2f}%\n")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-only', action='store_true')
    parser.add_argument('--sweep-only', action='store_true')
    args = parser.parse_args()

    out = {}

    if not args.sweep_only:
        seeds, mu, sigma = run_multiseed()
        out['multiseed'] = {
            'runs': seeds,
            'mean_mare': round(mu, 4),
            'std_mare': round(sigma, 4),
        }

    if not args.seed_only:
        sweep = run_sweep()
        out['sweep'] = sweep

    out_path = ROOT / 'fdm_output' / 'experiment_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
