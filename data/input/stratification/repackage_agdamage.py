#!/usr/bin/env python3
"""repackage_agdamage.py — repackages AgDamage chips (Flood and Burnt hazards) into
sharded tar archives with an event-disjoint split: a cross-region **OOD hold-out**
carved out first, then a per-hazard, quantile-stratified train/val/test split on the
remainder ("dev"). This mirrors repackage_agdamage_ootd.ipynb exactly (same severity
aggregation, OOD-candidate ranking/eligibility, and quantile-stratified allocation)
so the notebook and this script never drift apart -- update_split.py imports the
same functions rather than reimplementing them.

Hazard-specific severity fields (e.g. flooded_crop_frac vs burned_crop_frac) are
resolved generically via SEVERITY_FIELD_CANDIDATES so the same script covers both.
"""
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
SEVERITY_FIELD_CANDIDATES = ["flooded_crop_frac", "burned_crop_frac"]
SEVERITY_FIELD_BY_HAZARD = {
    "flooded": "flooded_crop_frac",
    "burnt": "burned_crop_frac", "burned": "burned_crop_frac", "fire": "burned_crop_frac",
}
DEFAULT_CHIP_SIZE = 512
# Offline (no network needed at run time) Natural Earth 1:50m admin-0 country
# boundaries, used by infer_missing_country() below. Some AgDamage sources never
# got hand-geocoded (e.g. Flood's "groundsource" chips are only ~8% populated,
# vs. 100% for "dfo" and for all of Burnt) -- --ood-unit=country then silently
# carves nothing for a real region that just isn't in the sparse `country`
# column yet. See repackage_agdamage_ootd.ipynb's country-inference cell.
NE_COUNTRIES_SHP = Path(__file__).parent / "naturalearth" / "ne_50m_admin_0_countries.shp"
# Flat sidecar/QC fields worth carrying into manifest.parquet (nested blobs like
# imagery/provenance/phenology, and local author file paths, are deliberately
# excluded -- they don't serialize cleanly to parquet and aren't needed downstream).
MANIFEST_META_KEYS = (
    "rank", "source", "tier", "continent", "country", "place",
    "event_date_start", "event_date_end", "crs", "res_m", "size",
    "cropland_frac", "excluded_crop_frac", "flood_persistence_mean",
    "burn_severity_mean", "min_lon", "min_lat", "max_lon", "max_lat", "qc_verdict",
    "lat", "lon", "latitude", "longitude", "centroid_lon", "centroid_lat",
    "date", "date_pre", "date_post", "row", "col", "geometry", "bbox",
    "flooded_crop_px", "burned_crop_px", "has_s1",
)

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
    severity_class: str | None = None
    subset: str = "dev"; split: str = ""


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
        c.meta = meta


def chips_to_dataframe(chips) -> pd.DataFrame:
    """Chip objects -> one row per chip, with the flat MANIFEST_META_KEYS (plus
    core identifiers) pulled out of each chip's sidecar/QC metadata. This is the
    same shape of table repackage_agdamage_ootd.ipynb starts from (its `df`),
    so aggregate_events() etc. below work identically whether chips come from
    discover_chips() here or from an already-repackaged manifest.parquet (see
    update_split.py)."""
    keys = set(MANIFEST_META_KEYS) | set(SEVERITY_FIELD_CANDIDATES)
    rows = []
    for c in chips:
        row = {"chip_id": c.chip_id, "event_id": c.event_id, "hazard": c.hazard}
        for k in keys:
            if k in c.meta: row[k] = c.meta[k]
        # discover_chips() only keeps chips with all 5 modality files present
        # (including s1_pre/s1_post), so S1 is always available downstream here.
        row.setdefault("has_s1", True)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================== severity aggregation ==============================
# Mirrors repackage_agdamage_ootd.ipynb cells 54449df5 / 48e0c130.

def resolve_damage_frac(df: pd.DataFrame) -> pd.Series:
    """Per-chip damage fraction, hazard-aware: prefer an explicit `damage_frac`
    column, else the hazard's own field (flooded_crop_frac/burned_crop_frac via
    SEVERITY_FIELD_BY_HAZARD), else any SEVERITY_FIELD_CANDIDATES fallback."""
    def _row(row):
        if "damage_frac" in df.columns and pd.notna(row.get("damage_frac")):
            return row["damage_frac"]
        fld = SEVERITY_FIELD_BY_HAZARD.get(str(row.get("hazard", "")).lower())
        if fld and fld in df.columns and pd.notna(row.get(fld)):
            return row[fld]
        for f in SEVERITY_FIELD_CANDIDATES:
            if f in df.columns and pd.notna(row.get(f)):
                return row[f]
        return float("nan")
    return df.apply(_row, axis=1).astype(float)


def add_cropland_px(df: pd.DataFrame, default_size: int = DEFAULT_CHIP_SIZE) -> pd.Series:
    """Per-chip cropland pixel count -- the inverse-variance-weighting weight
    for event-level severity_ivw. Uses an explicit `cropland_px` column if
    present, else derives it from cropland_frac * size**2."""
    if "cropland_px" in df.columns:
        return df["cropland_px"].astype(float)
    size = df["size"].astype(float) if "size" in df.columns else float(default_size)
    frac = df["cropland_frac"].astype(float) if "cropland_frac" in df.columns else 0.0
    return (frac * (np.asarray(size, dtype=float) ** 2)).round()


def infer_missing_country(df: pd.DataFrame, shp_path: Path = NE_COUNTRIES_SHP) -> pd.DataFrame:
    """Fill missing `country` values from each chip's centroid (derived from
    min/max_lon/lat) via offline point-in-polygon lookup against a local
    Natural Earth 1:50m admin-0 countries shapefile, with a nearest-country
    fallback (in a projected CRS) for coastal points that fall outside every
    polygon at 50m resolution. No-ops (returns df unchanged) if `country` is
    already fully populated, there's no lon/lat to work with, or geopandas /
    the shapefile aren't available -- so this is always safe to call."""
    if "country" not in df.columns or df["country"].notna().all():
        return df
    if not {"min_lon", "max_lon", "min_lat", "max_lat"} <= set(df.columns):
        return df
    if not shp_path.exists():
        log.warning("Natural Earth shapefile not found at %s -- %d chips with missing "
                     "country will stay unresolved", shp_path, int(df["country"].isna().sum()))
        return df
    try:
        import geopandas as gpd
    except ImportError:
        log.warning("geopandas not installed -- %d chips with missing country will stay "
                     "unresolved", int(df["country"].isna().sum()))
        return df

    df = df.copy()
    need = df["country"].isna()
    clon = (df["min_lon"].astype(float) + df["max_lon"].astype(float)) / 2
    clat = (df["min_lat"].astype(float) + df["max_lat"].astype(float)) / 2
    need &= clon.notna() & clat.notna()
    if not need.any():
        return df

    countries = gpd.read_file(shp_path)[["NAME", "geometry"]]
    pts = gpd.GeoDataFrame({"idx": df.index[need]},
                            geometry=gpd.points_from_xy(clon[need], clat[need]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, countries, how="left", predicate="within").drop_duplicates("idx").set_index("idx")
    unmatched = joined[joined["NAME"].isna()]
    if len(unmatched):
        pts_m = pts.set_index("idx").to_crs(3857)
        nearest = gpd.sjoin_nearest(pts_m.loc[unmatched.index, ["geometry"]], countries.to_crs(3857), how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[nearest.index, "NAME"] = nearest["NAME"]

    n_filled = int(joined["NAME"].notna().sum())
    df.loc[joined.index, "country"] = df.loc[joined.index, "country"].fillna(joined["NAME"])
    log.info("Inferred country for %d/%d chips missing it (offline Natural Earth lookup)",
              n_filled, int(need.sum()))
    return df


def require_country_inference_ready(ood_unit: str, ood_regions: dict) -> None:
    """Fail fast and loud, before any manifest/shard work starts, if a
    country-based OOD carve is requested but infer_missing_country() can't
    actually run. Most AgDamage sources (e.g. Flood's "groundsource" chips --
    ~92% of them) have `country` populated for only a small fraction of
    events; without offline lat/lon inference a real region can silently
    match 0 events (carve_ood() then raises, but three calls deep and with a
    confusingly short "known countries" list). continent doesn't need this --
    it's fully populated in every AgDamage source seen so far."""
    if ood_unit != "country" or not (ood_regions or {}):
        return
    if not NE_COUNTRIES_SHP.exists():
        raise SystemExit(
            f"--ood-unit=country with --ood-regions requested, but the Natural Earth "
            f"shapefile is missing at {NE_COUNTRIES_SHP} -- country inference for "
            f"sparsely-geocoded chips can't run."
        )
    try:
        import geopandas  # noqa: F401
    except ImportError:
        raise SystemExit(
            "--ood-unit=country with --ood-regions requested, but geopandas isn't "
            "importable in this Python -- offline lat/lon country inference "
            "(infer_missing_country()) can't run, so a real region can silently match "
            "0 events (e.g. Flood's 'groundsource' chips are only ~8% geocoded without "
            "it). Run this script with the project's 'lora' conda env instead of the "
            "system python3, e.g.:\n"
            "  ~/.conda/envs/lora/bin/python update_split.py ...\n"
            "(same env used for training -- see requirements.txt's note on webdataset/einops)."
        )


def aggregate_events(df: pd.DataFrame, min_cropland_frac: float = 0.0) -> pd.DataFrame:
    """Chip-level table -> one row per (event_id, hazard): n_chips,
    severity_mean (plain mean of chip damage_frac), severity_ivw
    (cropland-pixel-weighted mean), centroid_lon/lat (mean of chip centroids,
    derived from min/max_lon/lat when no explicit centroid column exists), and
    country/continent (first non-null value seen for the event)."""
    d = infer_missing_country(df).copy()
    d["damage_frac"] = resolve_damage_frac(d)
    d["cropland_px"] = add_cropland_px(d)
    if "cropland_frac" in d.columns:
        d = d[d["cropland_frac"].astype(float) >= min_cropland_frac]
    d = d.dropna(subset=["damage_frac"])
    d["w"] = d["cropland_px"].clip(lower=0)
    d["wf"] = d["w"] * d["damage_frac"]
    if "centroid_lon" not in d.columns and {"min_lon", "max_lon"} <= set(d.columns):
        d["centroid_lon"] = (d["min_lon"].astype(float) + d["max_lon"].astype(float)) / 2
        d["centroid_lat"] = (d["min_lat"].astype(float) + d["max_lat"].astype(float)) / 2
    for c in ("centroid_lon", "centroid_lat"):
        if c not in d.columns: d[c] = np.nan

    aggd = {"n_chips": ("damage_frac", "size"), "severity_mean": ("damage_frac", "mean"),
            "_ws": ("w", "sum"), "_wf": ("wf", "sum"),
            "centroid_lon": ("centroid_lon", "mean"), "centroid_lat": ("centroid_lat", "mean")}
    for col in ("country", "continent"):
        if col in d.columns: aggd[col] = (col, "first")
    events = d.groupby(["event_id", "hazard"], as_index=False).agg(**aggd)
    events["severity_ivw"] = np.where(events["_ws"] > 0, events["_wf"] / events["_ws"], np.nan)
    return events.drop(columns=["_ws", "_wf"])


# ============================== cross-region OOD hold-out ==============================
# Mirrors repackage_agdamage_ootd.ipynb cells 4ca325a4 (research) / 5fc5829b (carve).

def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, np.asarray(lon2, float), np.asarray(lat2, float)])
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(d))


def _wasserstein1d(a, b, n=1000):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0: return float("nan")
    q = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def rank_ood_candidates(events: pd.DataFrame, ood_unit: str, severity_col: str,
                         min_events: int, buffer_km: float, max_frac: float = 0.20) -> pd.DataFrame:
    """Rank each (hazard, region) as a cross-region OOD hold-out candidate.
    Scores: size (n_events/frac_events), severity-representativeness
    (Wasserstein distance of the region's severity distribution vs the rest of
    the hazard -- smaller = spans a similar minor->catastrophic range), and
    spatial isolation (min haversine km from any event in the region to any
    event outside it -- larger = cleaner separation). `eligible` requires
    n_events>=min_events, frac_events<=max_frac, isolation_km>=buffer_km."""
    if ood_unit not in events.columns or not len(events):
        return pd.DataFrame()
    rows = []
    for hz, dh in events.groupby("hazard"):
        tot = len(dh)
        for region, grp in dh.groupby(ood_unit):
            rest = dh[dh[ood_unit] != region]
            iso = float("nan")
            gg = grp.dropna(subset=["centroid_lon", "centroid_lat"])
            rr = rest.dropna(subset=["centroid_lon", "centroid_lat"])
            if len(gg) and len(rr):
                iso = float(np.min([
                    _haversine_km(r.centroid_lon, r.centroid_lat,
                                  rr["centroid_lon"].values, rr["centroid_lat"].values).min()
                    for r in gg.itertuples()
                ]))
            rows.append(dict(
                hazard=hz, region=region, n_events=len(grp), frac_events=len(grp) / tot,
                n_chips=int(grp["n_chips"].sum()), sev_min=float(grp[severity_col].min()),
                sev_max=float(grp[severity_col].max()),
                repr_wass=_wasserstein1d(grp[severity_col], rest[severity_col]), isolation_km=iso,
            ))
    cand = pd.DataFrame(rows)
    if len(cand):
        cand["eligible"] = ((cand.n_events >= min_events) & (cand.frac_events <= max_frac)
                             & (cand.isolation_km.fillna(0) >= buffer_km))
        cand = cand.sort_values(["hazard", "eligible", "repr_wass"], ascending=[True, False, True])
    return cand


def carve_ood(events: pd.DataFrame, ood_unit: str, ood_regions: dict) -> pd.DataFrame:
    """Tag events in the chosen OOD region(s) per hazard as subset="ood_holdout";
    everything else "dev". `ood_regions`: {hazard: [region, ...]}, hazard match
    is case-insensitive. Run *before* the train/val/test split so held-out
    events never leak into it.

    Raises if a requested hazard/region pair matches zero events, rather than
    silently leaving everything in "dev" -- a region name that doesn't appear
    in `ood_unit` (e.g. a typo, or a real region not yet present in a sparsely
    geocoded `country` column -- see infer_missing_country()) used to produce
    an empty ood_holdout with no shard and no error at all."""
    events = events.copy()
    events["subset"] = "dev"
    for hz, regs in (ood_regions or {}).items():
        if not regs: continue
        if ood_unit not in events.columns:
            raise ValueError(f"--ood-unit={ood_unit!r} but events has no {ood_unit!r} column "
                              f"-- cannot carve OOD for hazard {hz!r}")
        hz_mask = events["hazard"].astype(str).str.lower() == str(hz).lower()
        m = hz_mask & events[ood_unit].isin(list(regs))
        if not m.any():
            available = sorted(events.loc[hz_mask, ood_unit].dropna().unique().tolist())
            shown = available[:40]
            raise ValueError(
                f"--ood-regions {regs!r} for hazard {hz!r} ({ood_unit}) matched 0 of "
                f"{int(hz_mask.sum())} '{hz}' events -- refusing to silently carve nothing. "
                f"Known {ood_unit} values for '{hz}': {shown}"
                + (f" ... ({len(available) - 40} more)" if len(available) > 40 else "")
            )
        events.loc[m, "subset"] = "ood_holdout"
    return events


# ============================== quantile-stratified split ==============================
# Mirrors repackage_agdamage_ootd.ipynb cell da2ffeee.

def quantile_stratified_split(events: pd.DataFrame, severity_col: str, k: int,
                               ratios, seed: int) -> pd.DataFrame:
    """Event-atomic, per-hazard quantile-stratified train/val/test split of the
    `dev` subset: cut each hazard's dev events into `k` quantile bins on
    severity_col, then seeded-shuffle + ratio-slice events within each
    (hazard, bin) stratum. `ood_holdout` events (subset != "dev") are left
    alone and get split="ood_holdout"."""
    rng = np.random.default_rng(seed)
    events = events.copy()
    events["severity_class"] = pd.Series([None] * len(events), index=events.index, dtype=object)
    events["split"] = pd.Series([None] * len(events), index=events.index, dtype=object)

    dev = events[events["subset"] == "dev"]
    for hz, dh in dev.groupby("hazard"):
        s = dh[severity_col]
        try:
            labels = pd.qcut(s, k, labels=[f"q{i + 1}" for i in range(k)], duplicates="drop")
        except ValueError:
            # Too few distinct values for k bins (tiny/near-constant hazard) --
            # fall back to a single stratum rather than erroring out.
            labels = pd.Series(["q1"] * len(s), index=s.index, dtype=object)
        events.loc[dh.index, "severity_class"] = labels.astype(object).values

        tmp = dh.assign(_cls=labels.values)
        for cls, grp in tmp.groupby("_cls", observed=True):
            ids = grp.index.tolist(); rng.shuffle(ids)
            n = len(ids)
            n_tr = int(round(n * ratios[0])); n_va = int(round(n * ratios[1])); n_te = n - n_tr - n_va
            if n_te < 0: n_va += n_te; n_te = 0
            for i, ev_idx in enumerate(ids):
                events.loc[ev_idx, "split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    events.loc[events["subset"] == "ood_holdout", "split"] = "ood_holdout"
    return events


def joint_stratified_split(events: pd.DataFrame, severity_col: str, k: int, ratios, seed: int,
                            geo_unit: str = "continent", geo_min_events: int = 30) -> pd.DataFrame:
    """Event-atomic, per-hazard split of the `dev` subset stratified on a JOINT
    key: `k` severity-quantile bins crossed with a coarse `geo_unit` bucket
    (values with fewer than `geo_min_events` dev events folded into a single
    "Other" pool, so no joint cell degenerates to 1-2 events). Mirrors
    quantile_stratified_split()'s mechanics exactly (seeded-shuffle +
    ratio-slice per stratum) but keys strata by (geo_bucket, severity_class)
    instead of severity_class alone -- validated against this dataset in
    stratification_by_joint.ipynb: single-axis stratification (severity-only
    or a fine spatial grid) always leaves the *other* axis to chance, and on
    ~1-1.5k dev events that's a measurable risk, not a theoretical one (e.g.
    Burnt's severity-only split drew a train/test longitude Wasserstein of
    5.40 purely by chance). Coarse geography (continent, not a fine grid) is
    deliberate: fine cells are small enough that spreading them across
    train/val/test risks putting spatially-adjacent events on opposite sides
    of a split -- the autocorrelation-leakage failure mode Kattenborn et al.
    2022 measured inflating scores up to 28%. `ood_holdout` events (subset !=
    "dev") are left alone and get split="ood_holdout"."""
    rng = np.random.default_rng(seed)
    events = events.copy()
    events["severity_class"] = pd.Series([None] * len(events), index=events.index, dtype=object)
    events["geo_bucket"] = pd.Series([None] * len(events), index=events.index, dtype=object)
    events["split"] = pd.Series([None] * len(events), index=events.index, dtype=object)

    dev = events[events["subset"] == "dev"]
    for hz, dh in dev.groupby("hazard"):
        s = dh[severity_col]
        try:
            sev_labels = pd.qcut(s, k, labels=[f"q{i + 1}" for i in range(k)], duplicates="drop")
        except ValueError:
            sev_labels = pd.Series(["q1"] * len(s), index=s.index, dtype=object)
        events.loc[dh.index, "severity_class"] = sev_labels.astype(object).values

        if geo_unit in dh.columns:
            counts = dh[geo_unit].value_counts()
            small = counts[counts < geo_min_events].index.tolist()
            geo_labels = dh[geo_unit].where(~dh[geo_unit].isin(small), "Other")
        else:
            geo_labels = pd.Series(["all"] * len(dh), index=dh.index, dtype=object)
        events.loc[dh.index, "geo_bucket"] = geo_labels.astype(object).values

        tmp = dh.assign(_sev=sev_labels.values, _geo=geo_labels.values)
        for (_geo, _sev), grp in tmp.groupby(["_geo", "_sev"], observed=True):
            ids = grp.index.tolist(); rng.shuffle(ids)
            n = len(ids)
            n_tr = int(round(n * ratios[0])); n_va = int(round(n * ratios[1])); n_te = n - n_tr - n_va
            if n_te < 0: n_va += n_te; n_te = 0
            for i, ev_idx in enumerate(ids):
                events.loc[ev_idx, "split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    events.loc[events["subset"] == "ood_holdout", "split"] = "ood_holdout"
    return events


def bin_ranges(events: pd.DataFrame, severity_col: str, k: int) -> pd.DataFrame:
    """[hazard, severity_class, low, high, n_events] -- what each quantile
    stratum spans, on `dev` events only. Mirrors the notebook's bin_ranges()."""
    rows = []
    dev = events[events["subset"] == "dev"]
    for hz, dh in dev.groupby("hazard"):
        s = dh[severity_col]
        try:
            _, edges = pd.qcut(s, k, retbins=True, duplicates="drop")
        except ValueError:
            continue
        labels = [f"q{i + 1}" for i in range(len(edges) - 1)]
        counts = pd.qcut(s, k, labels=labels, duplicates="drop").value_counts()
        for i, lab in enumerate(labels):
            rows.append(dict(hazard=hz, severity_class=lab, low=float(edges[i]), high=float(edges[i + 1]),
                              n_events=int(counts.get(lab, 0))))
    return pd.DataFrame(rows)


def apply_split_to_chips(chip_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Left-join each chip to its event's severity_mean/severity_ivw/
    severity_class/subset/split (plus geo_bucket, when the split came from
    joint_stratified_split()). Mirrors the notebook's final `manifest =
    df.merge(events[keep], on="event_id", how="left")`. Drops any same-named
    columns already on chip_df first (e.g. update_split.py's input manifest
    may already carry a stale `split`/`severity_class` from a prior run) so
    the merge overwrites rather than suffixing them into split_x/split_y."""
    keep = ["event_id", "severity_mean", "severity_ivw", "severity_class", "subset", "split"]
    if "geo_bucket" in events.columns:
        keep.append("geo_bucket")
    chip_df = chip_df.drop(columns=[c for c in keep[1:] if c in chip_df.columns])
    return chip_df.merge(events[keep], on="event_id", how="left")


def split_summary(chip_df: pd.DataFrame, events: pd.DataFrame, ood_unit: str,
                   ood_regions: dict, ratios, k: int, severity_col: str, seed: int) -> dict:
    """Mirrors the notebook's split_summary.json (cell 6620eef6): per-hazard
    event/chip counts by split, which regions were held out, severity-class
    counts, a train/val/test Wasserstein balance check, and the event-atomic
    leakage assertion (every event maps to exactly one split, chips included)."""
    summary = {"seed": seed, "ratios": list(ratios), "k": k, "severity_col": severity_col,
               "ood_unit": ood_unit, "ood_regions": ood_regions or {}, "by_hazard": {}, "leakage_check": {}}
    for hz, dh in events.groupby("hazard"):
        ch = chip_df[chip_df["hazard"] == hz]
        parts = {sp: dh[dh["split"] == sp][severity_col].values for sp in ("train", "val", "test")}
        summary["by_hazard"][hz] = {
            "events": {sp: int((dh["split"] == sp).sum()) for sp in ("train", "val", "test", "ood_holdout")},
            "chips": {sp: int((ch["split"] == sp).sum()) for sp in ("train", "val", "test", "ood_holdout")},
            "ood_regions": sorted(dh[dh["subset"] == "ood_holdout"][ood_unit].dropna().unique().tolist())
                            if ood_unit in dh.columns else [],
            "severity_class_counts": {sp: dh[dh["split"] == sp]["severity_class"].value_counts().to_dict()
                                       for sp in ("train", "val", "test")},
            **({"geo_bucket_counts": {sp: dh[dh["split"] == sp]["geo_bucket"].value_counts().to_dict()
                                       for sp in ("train", "val", "test")}} if "geo_bucket" in dh.columns else {}),
            "wasserstein": {"train_vs_test": _wasserstein1d(parts["train"], parts["test"]),
                            "train_vs_val": _wasserstein1d(parts["train"], parts["val"]),
                            "val_vs_test": _wasserstein1d(parts["val"], parts["test"])},
        }
    ev_bad = events.groupby("event_id")["split"].nunique()
    ch_bad = chip_df.groupby("event_id")["split"].nunique()
    leaked = sorted(set(ev_bad[ev_bad > 1].index) | set(ch_bad[ch_bad > 1].index))
    summary["leakage_check"] = {"events_in_multiple_splits": leaked, "passed": len(leaked) == 0}
    return summary


# ============================== shard / manifest writing ==============================

def write_shards(chips, out, sps, severity_by_chip: dict):
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
                   "subset": c.subset, "split": c.split, "severity_class": c.severity_class,
                   **severity_by_chip.get(c.chip_id, {}), **c.meta}
            _add(tar, f"{key}.json", json.dumps(rec, default=str).encode())
            c.meta["__shard__"] = shard_name
        if tar: tar.close()
        log.info("[%s] %d chips -> %d shard(s)", split, len(cs), idx + 1)


def _add(tar, arcname, data):
    info = tarfile.TarInfo(arcname); info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_manifest(chip_df: pd.DataFrame, events: pd.DataFrame, out: Path, ratios, k: int,
                    severity_col: str, ood_unit: str, ood_regions: dict, seed: int,
                    ood_candidates: pd.DataFrame | None = None):
    df = chip_df.sort_values(["hazard", "subset", "split", "event_id", "chip_id"])
    df.to_parquet(out / "manifest.parquet", index=False)
    df.to_csv(out / "manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    ranges = bin_ranges(events, severity_col, k)
    if len(ranges): ranges.to_csv(out / "severity_bin_ranges.csv", index=False)
    if ood_candidates is not None and len(ood_candidates):
        ood_candidates.to_csv(out / "ood_candidates.csv", index=False)

    summary = split_summary(df, events, ood_unit, ood_regions, ratios, k, severity_col, seed)
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not summary["leakage_check"]["passed"]:
        log.error("LEAKAGE: %s", summary["leakage_check"]["events_in_multiple_splits"][:10])
        raise SystemExit(2)
    log.info("Event-atomic check PASSED (%d events, %d chips)", events["event_id"].nunique(), len(df))


# usage: python repackage_agdamage.py --src AgDamage_raw --out AgDamage_v2 \
#          --samples-per-shard 32 --ood-unit country --ood-regions '{"Flood": ["Vietnam"]}'
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
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
                     help='JSON, e.g. \'{"Flood": ["Vietnam"], "Burnt": ["Benin"]}\'. '
                          "Empty = nothing held out; the split uses all events.")
    ap.add_argument("--ood-min-events", type=int, default=30)
    ap.add_argument("--ood-buffer-km", type=float, default=50.0)
    ap.add_argument("--ood-max-frac", type=float, default=0.20)
    args = ap.parse_args(argv)
    if abs(sum(args.ratios) - 1.0) > 1e-6: ap.error("--ratios must sum to 1.0")
    try:
        ood_regions = json.loads(args.ood_regions)
    except json.JSONDecodeError as e:
        ap.error(f"--ood-regions is not valid JSON: {e}")
    require_country_inference_ready(args.ood_unit, ood_regions)
    args.out.mkdir(parents=True, exist_ok=True)
    severity_col = f"severity_{args.severity_col}"

    chips, dropped = discover_chips(args.src)
    if not chips: log.error("no chips"); return 1
    attach_metadata(chips, args.src)

    # Repackage each hazard into its own top-level output folder (out/<Hazard>/...),
    # mirroring the AgDamage_v2 layout -- hazards are never merged into one shard
    # tree, since severity/OOD/split are computed and consumed per hazard.
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

        chip_df = chips_to_dataframe(hazard_chips)
        events = aggregate_events(chip_df, args.min_cropland_frac)
        cand = rank_ood_candidates(events, args.ood_unit, severity_col,
                                    args.ood_min_events, args.ood_buffer_km, args.ood_max_frac)
        if len(cand):
            n_eligible = int(cand["eligible"].sum())
            log.info("[%s] OOD candidates: %d regions ranked, %d eligible (see ood_candidates.csv)",
                      hazard, len(cand), n_eligible)
        events = carve_ood(events, args.ood_unit, ood_regions)
        events = quantile_stratified_split(events, severity_col, args.k, tuple(args.ratios), args.seed)
        chip_df = apply_split_to_chips(chip_df, events)

        split_of = dict(zip(chip_df["chip_id"], chip_df["split"]))
        subset_of = dict(zip(chip_df["chip_id"], chip_df["subset"]))
        sevcls_of = dict(zip(chip_df["chip_id"], chip_df["severity_class"]))
        for c in hazard_chips:
            c.split = split_of[c.chip_id]; c.subset = subset_of[c.chip_id]
            c.severity_class = sevcls_of[c.chip_id]

        severity_by_chip = chip_df.set_index("chip_id")[["severity_mean", "severity_ivw"]].to_dict("index")
        write_shards(hazard_chips, hazard_out, args.samples_per_shard, severity_by_chip)
        shard_of = {c.chip_id: c.meta.get("__shard__", "") for c in hazard_chips}
        chip_df["shard"] = chip_df["chip_id"].map(shard_of)
        write_manifest(chip_df, events, hazard_out, tuple(args.ratios), args.k, severity_col,
                        args.ood_unit, ood_regions, args.seed, ood_candidates=cand)

    log.info("DONE -> %s", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
