import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_tide_values(trace_files, boundary_trim):
    values = []
    stats = []
    for path in trace_files:
        df = pd.read_csv(path)
        if "tide" not in df.columns:
            continue
        tide = pd.to_numeric(df["tide"], errors="coerce").to_numpy()
        tide = tide[np.isfinite(tide)]
        raw_count = int(tide.shape[0])
        if raw_count == 0:
            stats.append({"file": str(path), "raw": 0, "kept": 0})
            continue
        if boundary_trim > 0 and raw_count > 2 * boundary_trim:
            tide = tide[boundary_trim:-boundary_trim]
        kept_count = int(tide.shape[0])
        values.append(tide)
        stats.append({"file": str(path), "raw": raw_count, "kept": kept_count})
    if len(values) == 0:
        return np.array([], dtype=np.float64), stats
    return np.concatenate(values, axis=0), stats


def main():
    parser = argparse.ArgumentParser(description="Calibrate DP-TIDE threshold q_hat from nominal successful traces.")
    parser.add_argument("--trace_dir", type=str, required=True, help="Directory containing tide_trace_episode*.csv")
    parser.add_argument("--pattern", type=str, default="tide_trace_episode*.csv", help="Glob pattern for trace files.")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Conformal tail probability. q_hat is (1-alpha)-quantile.")
    parser.add_argument("--boundary_trim", type=int, default=1, help="Trim this many points at both start/end per episode.")
    parser.add_argument("--out", type=str, required=True, help="Output threshold json path.")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    files = sorted(trace_dir.glob(args.pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No trace files found in {trace_dir} with pattern {args.pattern}")

    values, file_stats = collect_tide_values(files, boundary_trim=args.boundary_trim)
    if values.size == 0:
        raise RuntimeError("No valid tide values found after filtering and trimming.")

    q_hat = float(np.quantile(values, 1.0 - args.alpha, method="higher"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "q_hat": q_hat,
        "alpha": float(args.alpha),
        "boundary_trim": int(args.boundary_trim),
        "n_files": int(len(files)),
        "n_values": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "source_pattern": args.pattern,
        "files": file_stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Calibrated q_hat={q_hat:.8f}")
    print(f"Saved threshold json: {out_path}")


if __name__ == "__main__":
    main()
