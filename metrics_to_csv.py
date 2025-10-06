import re
import numpy as np
import pandas as pd
from pathlib import Path

# Root containing all run folders like: results/<experiment>_seed1
ROOT = Path("results")

# Metrics to read (expecting .npz with 'arr_0' OR .npy)
METRICS = ['3d_hd95', '3d_dice', '3d_hd', '3d_assd', '3d_nsd', '3d_jaccard']

# Regex to split "<experiment_base>_seed<id>"
SEED_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")

rows = []           # one row per run
per_exp_values = {} # {base: {metric: [np.array per run]}}

# Find only seed run folders directly under results/
for run_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    m = SEED_RE.match(run_dir.name)
    if not m:
        # Not a run folder like "*_seedX" → skip
        continue

    base = m.group("base")
    seed = m.group("seed")
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        # The only path we care about; skip others
        print(f"Skipping (no metrics): {metrics_dir}")
        continue

    # Prepare row for this run
    row = {"experiment": base, "run": run_dir.name, "seed": int(seed)}

    # Load metrics for this run
    for metric in METRICS:
        npz_path = metrics_dir / f"{metric}.npz"
        npy_path = metrics_dir / f"{metric}.npy"

        if npz_path.exists():
            with np.load(npz_path) as data:
                key = list(data.keys())[0]  # Load the first (and likely only) array
                arr = data[key]
        elif npy_path.exists():
            arr = np.load(npy_path)
        else:
            print(f"Missing {metric} in {metrics_dir}")
            continue

        # Save per-class values in columns
        for i, v in enumerate(arr, start=1):
            row[f"{metric}_class{i}"] = float(v)

        # Accumulate for per-experiment mean/std
        per_exp_values.setdefault(base, {}).setdefault(metric, []).append(arr)

    rows.append(row)

# Dataframe with one row per run
df_runs = pd.DataFrame(rows).sort_values(by=["experiment", "seed"]).reset_index(drop=True)

# Build per-experiment mean/std rows
summary_rows = []
for base, metric_dict in per_exp_values.items():
    mean_row = {"experiment": base, "run": "mean", "seed": np.nan}
    std_row  = {"experiment": base, "run": "std",  "seed": np.nan}

    for metric, arrays in metric_dict.items():
        if not arrays:
            continue
        stacked = np.stack(arrays, axis=0)   # shape: (num_runs, num_classes)
        mean_vals = stacked.mean(axis=0)
        std_vals  = stacked.std(axis=0)

        for i, (m, s) in enumerate(zip(mean_vals, std_vals), start=1):
            mean_row[f"{metric}_class{i}"] = float(m)
            std_row[f"{metric}_class{i}"]  = float(s)

    summary_rows.append(mean_row)
    summary_rows.append(std_row)

df_summary = pd.DataFrame(summary_rows)

# Combine and save
df_out = pd.concat([df_runs, df_summary], ignore_index=True)
df_out.to_csv("all_metrics_runs_and_summary.csv", index=False)
print("Saved to all_metrics_runs_and_summary.csv")
