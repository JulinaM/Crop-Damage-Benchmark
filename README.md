# Crop Damage Benchmark
 Existing EO disaster datasets are fragmented by hazard, centered on building or urban damage (e.g., xBD, BRIGHT), or limited to flood and burn-scar extent with no link to cropland. The gap persists at the foundation-model level: across geospatial foundation model (GeoFM) benchmarks such as GEO-Bench, PANGAEA, and GEO-Bench-2, the only recurring disaster tasks are flood, burn-scar, and building-damage segmentation, leaving agricultural impact unrepresented. 
 We introduce CropDamage Benchmark, a multi-hazard, extent benchmark that connects observed disaster extent to cropland, and is architected so that severity (damage) can later be layered onto the same imagery. CropDamage Benchmark pairs bi-temporal (pre/post-event) Sentinel-1 SAR and Sentinel-2 optical imagery at 10 m, tiled into foundation-model-native 512x512 chips, providing pixel-wise three-class extent masks (unaffected / non-crops / damaged) and a derived cropland-intersection mask from USDA CDL and ESA WorldCover. 
 
—
 

## Structure

 

```

crop_damage/            # Core training / inference code (trainer, models, loaders)

configs/                # Experiment and model configs

slurm/                  # SLURM launch scripts for cluster training

dataset_construction/   # Pipeline for building the benchmark dataset

```

 

## Quick start

 

```bash

git clone https://github.com/JulinaM/Crop-Damage-Benchmark.git

cd Crop-Damage-Benchmark

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

 

# small experiment on 2 GPUs

torchrun --nproc_per_node=2 crop_damage/trainer.py -m ++train_loader=small

 

# scheduled training on SLURM

sbatch slurm/terramind.slurm

```

 

SLURM options: `DATA_SIZE=large|small` (default `large`), `SLURM_LOG_LEVEL=debug`.

 

## Dataset

 

The benchmark dataset is built from raw Sentinel-1/Sentinel-2 scenes and split into

Train/Validation/Test sets stratified by hazard and agroecological context. The link to full dataset is `https://huggingface.co/datasets/eadrah/AgDamage_Benchmark`


See [`dataset_construction/`](dataset_construction/) for the pipeline.

 

## Acknowledgements

 

Builds on the [TerraMind](https://github.com/IBM/terramind) foundation model and the

Copernicus Sentinel-1/2 missions; originated from research code in

[DamageMappingTerramind](https://github.com/JulinaM/DamageMappingTerramind).

 

## License

 

MIT — see [`LICENSE`](LICENSE).

