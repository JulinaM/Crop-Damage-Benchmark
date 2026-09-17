#!/usr/bin/env python3
"""update_split.py — adds `subset`/`split` columns to an already-repackaged
AgDamage manifest (event-atomic, cross-region OOD hold-out + quantile-stratified
train/val/test on the remainder -- identical algorithm to
repackage_agdamage_ootd.ipynb and repackage_agdamage.py), and physically
re-shards the chip tars into shards/<split>/ subfolders (split now includes
"ood_holdout" alongside train/val/test) so that AgDamageShardDataset (which
hardcodes `hazard_dir/shards/<split>/<shard>`) can read the result with no
code changes.

Why this script exists: some AgDamage manifests (e.g. data/input/J_AgDamage/
AgDamage_Benchmark_v1/<Hazard>/) already carry the rich per-chip metadata
(source, tier, country, continent, cropland_frac, min/max_lon/lat, ...) that
repackage_agdamage_ootd.ipynb's severity/OOD logic needs, but either have no
`split` column yet or need it recomputed. Their shards are a flat
shards/<hazard>-NNNNNN.tar layout, each mixing chips from ~30 different events
(verified empirically) -- so a whole shard file can never simply be
renamed/moved into a split folder; every chip has to be re-extracted from its
old shard and re-packed into a new, split-scoped shard, one chip at a time.

The severity aggregation, OOD-candidate ranking/carving, and stratified split
are NOT reimplemented here -- they're imported from repackage_agdamage.py so
the notebook and both scripts stay in sync. Read that file's docstrings
(aggregate_events, rank_ood_candidates, carve_ood, quantile_stratified_split,
joint_stratified_split) for the full algorithm writeup. `--strata-mode`
selects which train/val/test stratification to use on the dev remainder:
"severity" (default, quantile bins on severity_col alone) or "joint" (severity
quantile crossed with a coarse `--geo-unit` bucket -- see
joint_stratified_split()'s docstring and stratification_by_joint.ipynb for why
this was added: single-axis stratification always leaves the *other* axis to
chance, which turned out to be a measurable, not just theoretical, risk on
this dataset's size).

This script **never writes into `--root`** -- output (manifest.parquet/csv,
split_summary.json, ood_candidates.csv, severity_bin_ranges.csv, and, with
--reshard, shards/<split>/*.tar) always goes to a **sibling** directory named
`<root>_resharded/` -- e.g. `--root .../AgDamage_Benchmark_v1` produces
`.../AgDamage_Benchmark_v1_resharded/<Hazard>/...`. This is true whether or
not `--reshard` is passed: without it, shards/ simply isn't touched at all
(so the resharded output's `shard` column still names the OLD flat
shards/<file>.tar, which won't resolve under AgDamageShardDataset's
`hazard_dir/shards/<split>/<shard>` lookup without --reshard). `--root` itself
is only ever read from. Promoting `<root>_resharded/<Hazard>/manifest.*` back
over the original `--root/<Hazard>/manifest.*` is a **separate, manual, and
deliberate** step this script never does on its own -- do that only if you
intend to lock in the new split for that hazard's production manifest.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from repackage_agdamage import (
    aggregate_events, apply_split_to_chips, carve_ood, joint_stratified_split,
    quantile_stratified_split, rank_ood_candidates, require_country_inference_ready, write_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("update_split")
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x


def load_manifest(hazard_dir: Path) -> pd.DataFrame:
    pq = hazard_dir / "manifest.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(hazard_dir / "manifest.csv")


def compute_split(df: pd.DataFrame, hazard: str, severity_col: str, k: int, ratios, seed: int,
                   ood_unit: str, ood_regions: dict, ood_min_events: int, ood_buffer_km: float,
                   ood_max_frac: float, min_cropland_frac: float = 0.0, strata_mode: str = "severity",
                   geo_unit: str = "continent", geo_min_events: int = 30):
    """Event-atomic cross-region OOD carve + stratified train/val/test split of
    the remainder, mirroring repackage_agdamage_ootd.ipynb end to end (see
    repackage_agdamage.aggregate_events/rank_ood_candidates/carve_ood for
    those steps). `strata_mode="severity"` (default) uses
    quantile_stratified_split (severity-quantile bins alone); "joint" uses
    joint_stratified_split (severity-quantile crossed with a coarse geo_unit
    bucket -- see that function's docstring for why). Returns
    (chip_df_with_subset_and_split, events_df, ood_candidates_df)."""
    df = df.copy()
    if "hazard" not in df.columns:
        df["hazard"] = hazard

    events = aggregate_events(df, min_cropland_frac)
    cand = rank_ood_candidates(events, ood_unit, severity_col, ood_min_events, ood_buffer_km, ood_max_frac)
    events = carve_ood(events, ood_unit, ood_regions)
    if strata_mode == "joint":
        events = joint_stratified_split(events, severity_col, k, ratios, seed, geo_unit, geo_min_events)
    else:
        events = quantile_stratified_split(events, severity_col, k, ratios, seed)
    df = apply_split_to_chips(df, events)
    return df, events, cand


def reshard(df: pd.DataFrame, hazard_dir: Path, out_hazard_dir: Path, sps: int) -> pd.DataFrame:
    """Re-pack each chip's tar members (all of them, whatever modalities it
    happens to have -- e.g. ~3% of Burnt chips have no s1_pre/s1_post member in
    their shard tar) out of its old flat shard (read from `hazard_dir/shards/`)
    and into `out_hazard_dir/shards/<split>/<split>-NNNNNN.tar`, sorted by
    chip_id within each split for a deterministic shard layout. `split` now
    includes "ood_holdout" alongside train/val/test -- handled the same way as
    any other split value here, since this just groups by whatever strings are
    in df["split"]. `hazard_dir` (the source) is only ever read, never written
    to -- `out_hazard_dir` is a separate tree (see module docstring) so the
    original can be deleted once the new one is verified. Returns df with
    `shard` rewritten to point at the new filename.
    """
    old_shard_of = dict(zip(df["chip_id"], df["shard"]))

    # Old tar handles + per-tar {chip_id: [TarInfo, ...]} index, both built
    # lazily on first touch and reused for the rest of the run -- each old
    # shard is scanned/opened at most once even though its chips end up
    # scattered across every split.
    open_old: dict[Path, tarfile.TarFile] = {}
    members_by_chip: dict[Path, dict] = {}

    def members_for(old_path: Path, chip_id: str):
        if old_path not in open_old:
            tf = tarfile.open(old_path, "r")
            open_old[old_path] = tf
            by_chip = defaultdict(list)
            for m in tf.getmembers():
                by_chip[m.name.split(".", 1)[0]].append(m)
            members_by_chip[old_path] = by_chip
        return open_old[old_path], members_by_chip[old_path][chip_id]

    new_shard_col = {}
    for split, g in df.groupby("split"):
        chip_ids = sorted(g["chip_id"])
        split_dir = out_hazard_dir / "shards" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        idx, out_tar, shard_name = -1, None, ""
        for i, chip_id in enumerate(tqdm(chip_ids, desc=f"reshard {split}")):
            if i % sps == 0:
                if out_tar: out_tar.close()
                idx += 1
                shard_name = f"{split}-{idx:06d}.tar"
                out_tar = tarfile.open(split_dir / shard_name, "w")
            old_path = hazard_dir / "shards" / old_shard_of[chip_id]
            tf, members = members_for(old_path, chip_id)
            for member in members:
                out_tar.addfile(member, tf.extractfile(member))
            new_shard_col[chip_id] = shard_name
        if out_tar: out_tar.close()
        log.info("[%s] %d chips -> %d shard(s)", split, len(chip_ids), idx + 1)

    for tf in open_old.values(): tf.close()
    df = df.copy()
    df["shard"] = df["chip_id"].map(new_shard_col)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                     default=Path(__file__).parent / "J_AgDamage" / "AgDamage_Benchmark_v1")
    ap.add_argument("--hazards", nargs="+", default=["Burnt"], choices=["Flood", "Flooded", "Burnt"])
    ap.add_argument("--ratios", nargs=3, type=float, default=(0.6, 0.2, 0.2),
                     help="train:val:test ratios applied to the dev remainder (post-OOD-carve).")
    ap.add_argument("--samples-per-shard", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4, help="number of severity quantile bins per hazard")
    ap.add_argument("--severity-col", choices=["ivw", "mean"], default="ivw",
                     help="severity_ivw (cropland-pixel-weighted) or severity_mean")
    ap.add_argument("--min-cropland-frac", type=float, default=0.0)
    ap.add_argument("--ood-unit", choices=["country", "continent"], default="country")
    ap.add_argument("--ood-regions", type=str, default="{}",
                     help='JSON, e.g. \'{"Burnt": ["South Africa"]}\'. Empty = nothing held out.')
    ap.add_argument("--ood-min-events", type=int, default=20)
    ap.add_argument("--ood-buffer-km", type=float, default=50.0)
    ap.add_argument("--ood-max-frac", type=float, default=0.12)
    ap.add_argument("--strata-mode", choices=["severity", "joint"], default="severity",
                     help="Train/val/test stratification on the dev remainder: 'severity' "
                          "(default, quantile bins on --severity-col alone) or 'joint' "
                          "(severity quantile crossed with a coarse --geo-unit bucket -- "
                          "see joint_stratified_split()'s docstring in repackage_agdamage.py).")
    ap.add_argument("--geo-unit", choices=["country", "continent"], default="continent",
                     help="Geography axis for --strata-mode=joint. Ignored otherwise.")
    ap.add_argument("--geo-min-events", type=int, default=30,
                     help="--strata-mode=joint only: geo_unit values with fewer dev events "
                          "than this are folded into a single 'Other' bucket.")
    ap.add_argument(
        "--reshard", action="store_true",
        help="Also physically re-pack shards into shards/<split>/ (see reshard()'s "
             "docstring). Off by default: it needs roughly another copy's worth of "
             "disk headroom (~60GB for Burnt). The new copy is written to a sibling "
             "directory, '<root>_resharded/' -- --root itself is never modified, so "
             "you can verify the new tree and then delete --root to reclaim space. "
             "Without --reshard, `shard` in the resharded-output manifest still names "
             "the OLD flat shards/<file>.tar, so AgDamageShardDataset.py's hardcoded "
             "hazard_dir/shards/<split>/<shard> lookup won't find them as-is.",
    )
    args = ap.parse_args(argv)
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        ap.error("--ratios must sum to 1.0")
    try:
        ood_regions = json.loads(args.ood_regions)
    except json.JSONDecodeError as e:
        ap.error(f"--ood-regions is not valid JSON: {e}")
    require_country_inference_ready(args.ood_unit, ood_regions)
    severity_col = f"severity_{args.severity_col}"
    out_root = args.root.parent / f"{args.root.name}_resharded" 

    
    for hazard in args.hazards:
        hazard_dir = args.root / hazard
        log.info("=== [%s] ===", hazard)
        df = load_manifest(hazard_dir)
        df, events, cand = compute_split(
            df, hazard, severity_col, args.k, tuple(args.ratios), args.seed,
            args.ood_unit, ood_regions, args.ood_min_events, args.ood_buffer_km,
            args.ood_max_frac, args.min_cropland_frac,
            strata_mode=args.strata_mode, geo_unit=args.geo_unit, geo_min_events=args.geo_min_events,
        )
        if len(cand):
            n_eligible = int(cand["eligible"].sum())
            log.info("[%s] OOD candidates: %d regions ranked, %d eligible (see ood_candidates.csv)",
                      hazard, len(cand), n_eligible)

        out_hazard_dir = out_root / hazard
        out_hazard_dir.mkdir(parents=True, exist_ok=True)
        if args.reshard:
            # out_hazard_dir = out_root / hazard
            # out_hazard_dir.mkdir(parents=True, exist_ok=True)
            df = reshard(df, hazard_dir, out_hazard_dir, args.samples_per_shard)
            write_manifest(df, events, out_hazard_dir, tuple(args.ratios), args.k, severity_col,
                            args.ood_unit, ood_regions, args.seed, ood_candidates=cand)
            log.info("[%s] resharded output written to %s -- verify it (leakage check above, "
                      "spot-load a few chips via AgDamageShardDataset), then %s can be deleted "
                      "to reclaim space.", hazard, out_hazard_dir, hazard_dir)
        else:
            log.warning(
                "[%s] --reshard not passed: shards/ stays flat, `shard` column still "
                "names shards/<file>.tar (including for the new ood_holdout bucket). "
                "AgDamageShardDataset.py won't find these without either re-running "
                "with --reshard (needs disk headroom) or a loader change to read "
                "hazard_dir/shards/<shard> directly.", hazard,
            )
            write_manifest(df, events, out_hazard_dir, tuple(args.ratios), args.k, severity_col,
                            args.ood_unit, ood_regions, args.seed, ood_candidates=cand)

    log.info("DONE")
    return 0

# hf download eadrah/AgDamage_Benchmark_v1 --repo-type=dataset --local-dir AgDamage_Benchmark_v2
# cd AgDamage_Benchmark_v2
#BURNT (severity-quantile, the default -- unchanged)
# USAGE (without reshard):      python3 update_split.py --root /fs/ess/PGS0413/AgDamage_Benchmark_v2/ --hazards Burnt --ood-regions '{"Burnt": ["South Africa"]}'
# USAGE (new tree):      python3 update_split.py --root /fs/ess/PGS0413/AgDamage_Benchmark_v2/ --hazards Burnt --ood-regions '{"Burnt": ["South Africa"]}' --reshard
#                       -> writes AgDamage_Benchmark_v2_resharded/Burnt/...

#BURNT (joint stratification -- severity x continent; see stratification_by_joint.ipynb)
# USAGE:      python3 update_split.py --root /fs/ess/PGS0413/AgDamage_Benchmark_v2/ --hazards Burnt --ood-regions '{"Burnt": ["South Africa"]}' --strata-mode joint --reshard
#                       -> writes AgDamage_Benchmark_v2_resharded/Burnt/... (same sibling-dir rule as above -- never touches --root)

#FLOOD
# USAGE (without reshard):      python3 update_split.py --root /fs/ess/PGS0413/AgDamage_Benchmark_v2/ --hazards Flooded --ood-regions '{"Flooded": ["Philippines"]}'
# USAGE (new tree):      python3 update_split.py --root /fs/ess/PGS0413/AgDamage_Benchmark_v2/ --hazards Flooded --ood-regions '{"Flooded": ["Philippines"]}' --reshard
#                       -> writes AgDamage_Benchmark_v2_resharded/Flooded/...
# USAGE (joint):      python3 update_split.py --root /fs/ess/PGS0413/AgDamage_Benchmark_v2/ --hazards Flooded --ood-regions '{"Flooded": ["Philippines"]}' --strata-mode joint --reshard

if __name__ == "__main__":
    sys.exit(main())
