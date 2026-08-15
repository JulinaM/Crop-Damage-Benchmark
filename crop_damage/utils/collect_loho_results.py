"""
Collect leave-one-hazard-out (LOHO) results across experiment runs.

Each Evaluator run writes {exp_dir}/{eval_name}/metrics.json (see
Evaluator._save_metrics_json). This script globs for those files under an
experiments root, pairs each run's in-distribution eval against its loho_*
eval(s) by experiment_name, and prints/saves a comparison table of macro IoU
(with bootstrap CI) so the generalization gap is visible at a glance instead
of buried in per-run text reports.

Usage:
    python -m crop_damage.utils.collect_loho_results data/experiments
    python -m crop_damage.utils.collect_loho_results data/experiments --out loho_report.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def _load_metrics(experiments_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in experiments_root.rglob("*/metrics.json"):
        with metrics_path.open() as handle:
            payload = json.load(handle)

        row = {
            "run_dir": str(metrics_path.parent.parent),
            "experiment_name": payload.get("experiment_name"),
            "eval_name": payload.get("eval_name"),
            "task": payload.get("task"),
            "encoder": payload.get("encoder"),
            "train_hazards": ",".join(payload.get("train_hazards", []) or []),
            "n_events": payload.get("n_events"),
            "n_chips": payload.get("n_chips"),
        }
        for name in ("IoU", "F1", "Accuracy", "Precision", "Recall"):
            row[f"macro_{name}"] = payload.get("macro", {}).get(name)
            ci = payload.get("macro_ci", {}).get(name, [None, None])
            row[f"macro_{name}_ci_low"] = ci[0]
            row[f"macro_{name}_ci_high"] = ci[1]
            row[f"micro_{name}"] = payload.get("micro", {}).get(name)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No metrics.json files found under {experiments_root}. "
            "Run an experiment first (Evaluator writes metrics.json per eval_loader)."
        )
    return pd.DataFrame(rows)


def build_loho_report(experiments_root: Path) -> pd.DataFrame:
    """
    One row per (experiment_name), with in-distribution macro IoU/F1 next to
    each loho_* eval's macro IoU/F1 and the generalization gap
    (in_distribution - loho), so cross-hazard generalization is a single
    number per run instead of separate scattered eval directories.
    """
    df = _load_metrics(experiments_root)

    in_dist = df[df["eval_name"].str.startswith("in_distribution")]
    loho = df[df["eval_name"].str.startswith("loho_")]

    report_rows = []
    for _, id_row in in_dist.iterrows():
        matches = loho[loho["experiment_name"] == id_row["experiment_name"]]
        if matches.empty:
            report_rows.append({
                "experiment_name": id_row["experiment_name"],
                "train_hazards": id_row["train_hazards"],
                "encoder": id_row["encoder"],
                "in_distribution_eval": id_row["eval_name"],
                "in_distribution_IoU": id_row["macro_IoU"],
                "in_distribution_IoU_CI": (id_row["macro_IoU_ci_low"], id_row["macro_IoU_ci_high"]),
                "loho_eval": None,
                "loho_IoU": None,
                "loho_IoU_CI": None,
                "generalization_gap_IoU": None,
            })
            continue

        for _, loho_row in matches.iterrows():
            gap = None
            if pd.notna(id_row["macro_IoU"]) and pd.notna(loho_row["macro_IoU"]):
                gap = id_row["macro_IoU"] - loho_row["macro_IoU"]
            report_rows.append({
                "experiment_name": id_row["experiment_name"],
                "train_hazards": id_row["train_hazards"],
                "encoder": id_row["encoder"],
                "in_distribution_eval": id_row["eval_name"],
                "in_distribution_IoU": id_row["macro_IoU"],
                "in_distribution_IoU_CI": (id_row["macro_IoU_ci_low"], id_row["macro_IoU_ci_high"]),
                "loho_eval": loho_row["eval_name"],
                "loho_IoU": loho_row["macro_IoU"],
                "loho_IoU_CI": (loho_row["macro_IoU_ci_low"], loho_row["macro_IoU_ci_high"]),
                "generalization_gap_IoU": gap,
            })

    return pd.DataFrame(report_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments_root", type=Path, help="Root dir containing experiment output dirs (e.g. data/experiments)")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV path to save the report to")
    args = parser.parse_args()

    report = build_loho_report(args.experiments_root)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(report.to_string(index=False))

    if args.out is not None:
        report.to_csv(args.out, index=False)
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
