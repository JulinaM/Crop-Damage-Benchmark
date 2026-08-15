# Crop Damage Benchmark
 Existing EO disaster datasets are fragmented by hazard, centered on building or urban damage (e.g., xBD, BRIGHT), or limited to flood and burn-scar extent with no link to cropland. The gap persists at the foundation-model level: across geospatial foundation model (GeoFM) benchmarks such as GEO-Bench, PANGAEA, and GEO-Bench-2, the only recurring disaster tasks are flood, burn-scar, and building-damage segmentation, leaving agricultural impact unrepresented. 
 We introduce CropDamage Benchmark, a multi-hazard, extent benchmark that connects observed disaster extent to cropland, and is architected so that severity (damage) can later be layered onto the same imagery. CropDamage Benchmark pairs bi-temporal (pre/post-event) Sentinel-1 SAR and Sentinel-2 optical imagery at 10 m, tiled into foundation-model-native 512x512 chips, providing pixel-wise three-class extent masks (unaffected / non-crops / damaged) and a derived cropland-intersection mask from USDA CDL and ESA WorldCover. 
 
—
 

## Structure

 

```

crop_damage/            # Core training / inference code (trainer, models, loaders)

configs/                # Experiment and model configs
Task A: Segmentation
Task B: Change Detection

slurm/                  # SLURM launch scripts for cluster training

dataset_construction/   # Pipeline for building the benchmark dataset

data/                   # input and logs from slurm and experiments

```

 

## Quick start

 

```bash

git clone https://github.com/JulinaM/Crop-Damage-Benchmark.git

cd Crop-Damage-Benchmark

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

 

# small experiment on 2 GPUs

torchrun --nproc_per_node=2 crop_damage/trainer.py -m ++train_loader=test

 

# scheduled training on SLURM

sbatch slurm/terramind.slurm

sbatch --export=DATA_SIZE=test,SLURM_LOG_LEVEL=debug slurm/terramind.slurm

```

 

SLURM options: `DATA_SIZE=large|test` (default `large`), `SLURM_LOG_LEVEL=debug`.

 

## Dataset

*(To do)*

The benchmark dataset is built from raw Sentinel-1/Sentinel-2 scenes and split into
Train/Validation/Test sets stratified by hazard and agroecological context.
Full dataset: **[huggingface.co/datasets/eadrah/AgDamage_Benchmark](https://huggingface.co/datasets/eadrah/AgDamage_Benchmark)**.

Each sample is a **chip**: a co-registered 512x512 tile at 10m resolution carrying pre/post-event
Sentinel-1 SAR + Sentinel-2 optical imagery and a 3-class damage label (unaffected / non-crop / damaged),
built by pairing the observed disaster extent (flood inundation, burn scar, ...) against cropland masks
(USDA CDL / ESA WorldCover). Two hazards are currently populated: **Flood** and **Burnt**.

See [`dataset_construction/`](dataset_construction/) for the raw-data collection pipeline, and
[`data/input/repackage_agdamage.py`](data/input/repackage_agdamage.py) for the repackaging script
described below.

### Raw layout (`AgDamage_raw`) and metadata

The as-downloaded HF dataset is organized one folder per disaster event:

```
AgDamage_raw/<Hazard>/
├── events_master.csv        # one row per event: bbox, date range, continent/country,
│                             # source (DFO/groundsource/...), tier, corroboration count,
│                             # gfm_flood_km2, n_chips, batch, ...
├── qc_chips.csv              # one row per chip: clear_frac, nodata_frac, edge_cc,
│                             # is_partial/is_white/is_corrupt/is_cloudy/is_speckle flags, ...
└── chips/<event_id>/
    ├── <chip_id>_label.tif
    ├── <chip_id>_s2_pre.tif / _s2_post.tif
    ├── <chip_id>_s1_pre.tif / _s1_post.tif
    └── <chip_id>.json        # per-chip sidecar: event provenance (source, tier, confirming
                               # agencies), event_date_start/end, crs, bbox (min/max lon/lat),
                               # cropland_frac, flooded_crop_frac / excluded_crop_frac,
                               # per-image acquisition date + clear_frac for each of the 4 rasters
```

### Repackaged distribution format (`AgDamage_v2` — used for training)

The raw layout is tens of thousands of loose files, which trips the HF API's rate limit.
`repackage_agdamage.py` converts it into **WebDataset shards + a Parquet manifest**:

```
AgDamage_v2/<Hazard>/
├── manifest.parquet / manifest.csv   # one row per chip: chip_id, event_id, hazard, split,
│                                      #   severity, severity_class, shard, + any lat/lon/date/crs
│                                      #   fields present in the chip's sidecar
├── split_summary.json                # per-split chip/event counts, severity_class histogram,
│                                      #   and an explicit event-leakage check (see below)
├── dropped_chips.csv                 # chips missing one of the 5 required rasters, with reason
└── shards/{train,val,test}/<split>-NNNNNN.tar
```

Inside a shard, one chip = 6 tar members sharing a key: `<chip_id>.{s1_pre,s1_post,s2_pre,s2_post,label}.tif`
+ `<chip_id>.json` (the manifest row, duplicated per-chip). `AgDamageShardDataset` (see
[`crop_damage/datasets/AgDamageShardDataset.py`](crop_damage/datasets/AgDamageShardDataset.py)) reads
these directly at train/eval time.

### Split & stratification logic

Splits are assigned **per event, not per chip** — every chip belonging to an event goes to exactly one
of train/val/test, so no event straddles splits (spatial autocorrelation between chips of the same event
would otherwise leak signal across the split boundary). `main()` runs the whole repackaging pipeline
**once per hazard**, writing each hazard's shards/manifest/split into its own `AgDamage_v2/<Hazard>/`
tree (never merged), so `assign_splits()` below is always called on a single hazard's chips at a time.
Concretely, `assign_splits()`:

1. Groups chips by `event_id`, and buckets each event into a severity class (`none` / `minor` / `moderate`
   / `severe` / `catastrophic`) from the mean of its chips' damage-fraction field — resolved generically
   via `SEVERITY_FIELD_CANDIDATES` (e.g. `flooded_crop_frac` for Flood, `burned_crop_frac` for Burnt), so
   the same bucketing logic works for both hazards.
2. Groups events into strata keyed by `(hazard, severity_class)`. Since `assign_splits()` is now always
   called with one hazard's chips already isolated, `hazard` is constant per call and this reduces in
   practice to stratifying by `severity_class` alone — the key is kept for robustness in case the function
   is ever called with a mixed-hazard chip list again.
3. Within each stratum, assigns whole events to train/val/test with a greedy largest-first bin-packing
   pass: events are sorted by descending chip count (random tiebreak on a fixed seed, for reproducibility),
   then walked in that order, each one assigned in full to whichever split is currently furthest below its
   target chip-count quota (`target - quota`, maximized). Quotas are targets over **chip volume**, not
   event count, so one very large event can't dominate a split. This is a greedy heuristic, not a globally
   optimal partition — per-stratum split sizes converge toward the configured ratios (default 70/15/15) but
   won't hit them exactly, especially for small strata (e.g. a severity class with very few events for that
   hazard), since an event's chips can't be split across quotas.
4. Writes `split_summary.json` with an explicit **event-disjointness check**: the script hard-fails
   (`SystemExit(2)`) if any event ends up in more than one split.

**Known gaps for the team to review before treating this as final:**
- The proximity-buffering step described in `CLAUDE.md` §4 (clustering events close in space *and* time
  into one super-group before splitting, with a spatial dead-zone between train/test) is not implemented
  in this script — only event-atomicity + severity-stratification + volume-balancing are.
- The numbers table previously shown here (Flood 160/38, Burnt 1000/315) predates three fixes: Burnt was
  being skipped entirely in `discover_chips()`, `main()` hard-truncated to `chips[:1000]`, and
  `SEVERITY_FIELD_CANDIDATES` had no burnt-specific field (so Burnt severity was `unknown` for every chip).
  All three are now fixed — re-run `repackage_agdamage.py` against the full raw dataset to regenerate
  `AgDamage_v2/` and replace this table with current numbers before relying on it.

## Evaluation Design

### Task A: Segmentation

Per-pixel damage classification (unaffected / non-crop / damaged) on each hazard, holding the rest of
the training protocol constant across models (decoder, patch size, augmentation, criterion, optimizer
budget — see `configs/segmentation/*.yaml`) so differences in results reflect the encoder, not the setup.
Three encoders × three hazard scopes = 9 configs:

| Encoder | Flood | Burnt | Pooled |
|---|---|---|---|
| Terramind | `terramind_flood.yaml` | `terramind_burnt.yaml` | `terramind_pooled.yaml` |
| U-Net (non-FM floor) | `unet_flood.yaml` | `unet_burnt.yaml` | `unet_pooled.yaml` |
| Prithvi | `prithvi_flood.yaml` | `prithvi_burnt.yaml` | `prithvi_pooled.yaml` |

- **Flood / Burnt** configs train on one hazard and evaluate two ways: **in-distribution** (same hazard's
  test split) and **leave-one-hazard-out (LOHO)** — zero-shot on the *other* hazard's test split, the
  headline cross-hazard generalization check.
- **Pooled** configs train on both hazards together and evaluate in-distribution on the pooled test split
  plus a per-hazard breakdown, so pooling gains/losses are visible per hazard rather than averaged away.
- **U-Net** is the from-scratch, non-foundation-model floor — run first, before treating any FM's number
  as meaningful, since it's the only way to tell whether a foundation model is actually adding value here.
- Reported metrics are macro-averaged **by event** (with bootstrap CIs) as the headline number, plus a
  micro (pixel-pooled) number — see `Evaluator._macro_and_micro_metrics`.

### Task B: Change Detection

*(Empty — not yet designed/implemented.)* Scaffolded at `configs/change_detection/` (see its README)
mirroring Task A's 3-encoder × 3-hazard-scope layout once ready.

### Future work: Finetune vs. Frozen FMs

`encoder.finetune` (true/false) already exists per-config, but every current config picks one setting
rather than running both. Per `CLAUDE.md` §5's fairness protocol, both should be reported for each
foundation model — frozen-encoder as the default probe of pretrained representation quality, full
fine-tune for the top-performing model(s) — since which one wins can flip the ranking between FMs.


## Acknowledgements

 

Builds on the [TerraMind](https://github.com/IBM/terramind) foundation model and the

Copernicus Sentinel-1/2 missions; originated from research code in

[DamageMappingTerramind](https://github.com/JulinaM/DamageMappingTerramind).

 

## License

 

MIT — see [`LICENSE`](LICENSE).

