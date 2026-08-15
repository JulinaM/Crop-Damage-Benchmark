#!/usr/bin/env python3
"""repackage_agdamage.py — repackages AgDamage chips (Flood and Burnt hazards) into
sharded tar archives with a stratified, event-disjoint train/val/test split.
Hazard-specific severity fields (e.g. flooded_crop_frac vs burned_crop_frac) are
resolved generically via SEVERITY_FIELD_CANDIDATES so the same script covers both."""
from __future__ import annotations
import argparse, csv, io, json, logging, sys, tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

MODALITY_SUFFIXES = {
    "s1_pre": "_s1_pre.tif", "s1_post": "_s1_post.tif",
    "s2_pre": "_s2_pre.tif", "s2_post": "_s2_post.tif", "label": "_label.tif",
}
SIDECAR_SUFFIX = ".json"
# Checked in order, per hazard's sidecar JSON, until one is present and numeric.
# Flood sidecars carry "flooded_crop_frac"; Burnt sidecars carry "burned_crop_frac".
# The remaining entries are generic fallbacks shared across hazards (or future ones)
# so a hazard folder with slightly different field naming still resolves a severity.
SEVERITY_FIELD_CANDIDATES = [
    "flooded_crop_frac", "burned_crop_frac", "damaged_crop_frac", "crop_damage_frac",
    "flood_frac", "burn_frac", "damage_frac", "label_frac",
]
SEVERITY_BINS = [(0.01, "none"), (0.10, "minor"), (0.30, "moderate"), (0.60, "severe"), (1.01, "catastrophic")]
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("repackage")
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **_): return x


@dataclass
class Chip:
    chip_id: str; event_id: str; hazard: str
    files: dict; sidecar: Path | None
    meta: dict = field(default_factory=dict)
    severity: float = float("nan"); severity_class: str = "unknown"; split: str = ""


def discover_chips(src: Path):
    chips, dropped = [], []
    hazard_dirs = [p for p in sorted(src.iterdir()) if p.is_dir() and (p / "chips").is_dir()]
    for hazard_dir in hazard_dirs:
        hazard = hazard_dir.name
        event_dirs = [p for p in sorted((hazard_dir / "chips").iterdir()) if p.is_dir()]
        log.info("[%s] scanning %d event folders", hazard, len(event_dirs))
        for ev_dir in event_dirs:
            event_id = ev_dir.name
            groups, sidecars = defaultdict(dict), {}
            for f in ev_dir.iterdir():
                if not f.is_file(): continue
                name, matched = f.name, False
                for mod, suf in MODALITY_SUFFIXES.items():
                    if name.endswith(suf):
                        groups[name[:-len(suf)]][mod] = f; matched = True; break
                if matched: continue
                if name.endswith(SIDECAR_SUFFIX):
                    sidecars[name[:-len(SIDECAR_SUFFIX)]] = f
            for chip_id, mods in groups.items():
                missing = [m for m in MODALITY_SUFFIXES if m not in mods]
                if missing:
                    dropped.append({"chip_id": chip_id, "event_id": event_id,
                                    "hazard": hazard, "missing": ",".join(missing)}); continue
                chips.append(Chip(chip_id, event_id, hazard, mods, sidecars.get(chip_id)))
    log.info("Discovered %d complete chips (%d dropped)", len(chips), len(dropped))
    return chips, dropped


def _find_severity(d):
    for k in SEVERITY_FIELD_CANDIDATES:
        if k in d and d[k] is not None:
            try: return float(d[k])
            except (TypeError, ValueError): pass
    return float("nan")


def classify(frac):
    if not np.isfinite(frac): return "unknown"
    for upper, name in SEVERITY_BINS:
        if frac < upper: return name
    return SEVERITY_BINS[-1][1]


def attach_metadata(chips, src):
    qc_by_chip = {}
    for hazard_dir in {c.hazard for c in chips}:
        qc_path = src / hazard_dir / "qc_chips.csv"
        if qc_path.exists():
            try:
                df = pd.read_csv(qc_path)
                id_col = next((c for c in df.columns
                               if c.lower() in ("chip_id", "chip", "id", "name")), None)
                if id_col:
                    for _, row in df.iterrows(): qc_by_chip[str(row[id_col])] = row.to_dict()
            except Exception as e:
                log.warning("qc read fail %s: %s", qc_path, e)
    for c in tqdm(chips, desc="metadata"):
        meta = {}
        if c.sidecar and c.sidecar.exists():
            try: meta.update(json.loads(c.sidecar.read_text()))
            except Exception: pass
        if c.chip_id in qc_by_chip:
            for k, v in qc_by_chip[c.chip_id].items(): meta.setdefault(k, v)
        c.meta = meta; c.severity = _find_severity(meta); c.severity_class = classify(c.severity)


def assign_splits(chips, ratios, seed):
    """Assign each *event* (not each chip) to train/val/test.

    Called once per hazard from main() — `chips` here is already filtered to a
    single hazard (e.g. all Flood chips, or all Burnt chips), because outputs are
    repackaged per hazard into separate out/<Hazard>/ trees (see main()). So the
    split, its shards, and its manifest are computed independently per hazard;
    nothing about ratios/quotas/stratum sizing below is shared across hazards.

    Splits are stratified at the event level for two reasons:
      1. Event-disjointness — every chip belonging to one event must land in the
         same split, otherwise near-duplicate/spatially-correlated chips from the
         same flood or fire event would leak across splits and inflate eval scores.
         (write_manifest() re-verifies this with an explicit leakage check.)
      2. Stratification — without it, a ratios-only random split could easily starve
         val/test of rare severity classes (e.g. "catastrophic"), since events vary
         a lot in both severity and chip count.

    Algorithm:
      a. Compute one severity value per event: the mean of its chips' severity
         fractions (NaNs ignored), then bucket that mean into a severity_class via
         classify() using the same SEVERITY_BINS as per-chip severity.
      b. Group events into strata keyed by (hazard, event_severity_class) — e.g.
         ("Flood", "moderate") or ("Burnt", "severe"). Since `chips` is already
         single-hazard, `hazard` is constant within a call, so in practice this
         only stratifies by severity_class — grouping events at severity level
         (rather than by chip count or raw fraction) so train/val/test each get a
         proportional mix of minor→catastrophic events instead of, say, all the
         catastrophic ones landing in train by chance. The `hazard` key is kept in
         the tuple (rather than dropped) so the function still works unmodified if
         it's ever called with a mixed-hazard chip list again.
      c. Within each stratum, run a greedy largest-first bin-packing pass:
         - Sort events by descending chip count (ties broken by a seeded RNG draw,
           so the result is deterministic for a given --seed but not alphabetical).
         - Compute each split's chip-count target for the stratum from `ratios`
           (e.g. 70/15/15) applied to the stratum's total chip count.
         - Walk events in that order; assign each whole event to whichever split is
           currently furthest below its target quota (target - quota, maximized).
           This is a greedy heuristic (not a globally optimal partition) but keeps
           per-stratum split sizes close to the requested ratios while guaranteeing
           every chip of an event goes to the same split.

    Because assignment happens per-stratum, the per-hazard dataset ratios are only
    approximate for small strata (e.g. a severity class with very few events for
    that hazard) — there just isn't enough granularity to hit 70/15/15 exactly
    when an event's chips can't be split. Larger strata converge closer to the
    target; hazards with fewer events (e.g. Burnt vs. Flood) will show more drift.
    """
    rng = np.random.default_rng(seed); names = ("train", "val", "test")
    ev_chips = defaultdict(list)
    for c in chips: ev_chips[c.event_id].append(c)
    ev_rows = []
    for ev, cs in ev_chips.items():
        fr = np.array([c.severity for c in cs], float); fin = fr[np.isfinite(fr)]
        ev_rows.append((ev, cs[0].hazard, classify(float(fin.mean()) if fin.size else float("nan")), len(cs)))
    strata = defaultdict(list)
    for ev, hz, sev, n in ev_rows: strata[(hz, sev)].append((ev, n))
    assignment = {}
    for stratum, evs in sorted(strata.items()):
        order = sorted(evs, key=lambda t: (-t[1], rng.random()))
        quota = {s: 0.0 for s in names}; total = sum(n for _, n in order)
        target = {s: r * total for s, r in zip(names, ratios)}
        for ev, n in order:
            pick = max(names, key=lambda s: target[s] - quota[s])
            assignment[ev] = pick; quota[pick] += n
    return assignment


def write_shards(chips, out, sps):
    by_split = defaultdict(list)
    for c in chips: by_split[c.split].append(c)
    for split, cs in by_split.items():
        cs.sort(key=lambda c: c.chip_id)
        sd = out / "shards" / split; sd.mkdir(parents=True, exist_ok=True)
        idx, tar, shard_name = -1, None, ""
        for i, c in enumerate(tqdm(cs, desc=f"shard {split}")):
            if i % sps == 0:
                if tar: tar.close()
                idx += 1; shard_name = f"{split}-{idx:06d}.tar"
                tar = tarfile.open(sd / shard_name, "w")
            key = c.chip_id
            for mod, path in c.files.items(): _add(tar, f"{key}.{mod}.tif", path.read_bytes())
            rec = {"chip_id": c.chip_id, "event_id": c.event_id, "hazard": c.hazard,
                   "split": c.split, "severity": c.severity,
                   "severity_class": c.severity_class, **c.meta}
            _add(tar, f"{key}.json", json.dumps(rec, default=str).encode())
            c.meta["__shard__"] = shard_name
        if tar: tar.close()
        log.info("[%s] %d chips -> %d shard(s)", split, len(cs), idx + 1)


def _add(tar, arcname, data):
    info = tarfile.TarInfo(arcname); info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_manifest(chips, out, ratios):
    rows = []
    for c in chips:
        row = {"chip_id": c.chip_id, "event_id": c.event_id, "hazard": c.hazard,
               "split": c.split, "severity": c.severity,
               "severity_class": c.severity_class, "shard": c.meta.get("__shard__", "")}
        for k in ("lat", "lon", "latitude", "longitude", "date", "date_pre",
                  "date_post", "crs", "row", "col", "geometry", "bbox"):
            if k in c.meta: row[k] = c.meta[k]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["hazard", "split", "event_id", "chip_id"])
    df.to_parquet(out / "manifest.parquet", index=False)
    df.to_csv(out / "manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {"ratios_target": dict(zip(("train", "val", "test"), ratios)),
               "n_chips": int(len(df)), "n_events": int(df["event_id"].nunique()),
               "by_split": {}, "leakage_check": {}}
    for split, g in df.groupby("split"):
        summary["by_split"][split] = {"chips": int(len(g)),
            "events": int(g["event_id"].nunique()), "chip_frac": round(len(g)/len(df), 4),
            "severity_class_counts": g["severity_class"].value_counts().to_dict()}
    ev_splits = df.groupby("event_id")["split"].nunique()
    leaked = ev_splits[ev_splits > 1].index.tolist()
    summary["leakage_check"] = {"events_in_multiple_splits": leaked, "passed": len(leaked) == 0}
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if leaked:
        log.error("LEAKAGE: %s", leaked[:10]); raise SystemExit(2)
    log.info("Event-disjointness PASSED (%d events, %d chips)", summary["n_events"], summary["n_chips"])


#usage: python ../repackage_agdamage.py --src AgDamage_raw --out AgDamage_v2 --samples-per-shard 32
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    ap.add_argument("--samples-per-shard", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if abs(sum(args.ratios) - 1.0) > 1e-6: ap.error("--ratios must sum to 1.0")
    args.out.mkdir(parents=True, exist_ok=True)
    chips, dropped = discover_chips(args.src)
    if not chips: log.error("no chips"); return 1
    attach_metadata(chips, args.src)

    # Repackage each hazard into its own top-level output folder (out/<Hazard>/...),
    # mirroring the AgDamage_v2 layout — hazards are never merged into one shard
    # tree, since splits/shards/manifests are computed and consumed per hazard.
    by_hazard = defaultdict(list)
    for c in chips: by_hazard[c.hazard].append(c)
    dropped_by_hazard = defaultdict(list)
    for d in dropped: dropped_by_hazard[d["hazard"]].append(d)

    for hazard, hazard_chips in sorted(by_hazard.items()):
        hazard_out = args.out / hazard
        hazard_out.mkdir(parents=True, exist_ok=True)
        log.info("=== [%s] %d chips ===", hazard, len(hazard_chips))
        if dropped_by_hazard.get(hazard):
            pd.DataFrame(dropped_by_hazard[hazard]).to_csv(hazard_out / "dropped_chips.csv", index=False)
        ev_split = assign_splits(hazard_chips, tuple(args.ratios), args.seed)
        for c in hazard_chips: c.split = ev_split[c.event_id]
        write_shards(hazard_chips, hazard_out, args.samples_per_shard)
        write_manifest(hazard_chips, hazard_out, tuple(args.ratios))

    log.info("DONE -> %s", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())