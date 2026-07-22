# Crop Damage Benchmark

 

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

Train/Validation/Test sets stratified by hazard and agroecological context. See

[`dataset_construction/`](dataset_construction/) for the pipeline.

 

## Acknowledgements

 

Builds on the [TerraMind](https://github.com/IBM/terramind) foundation model and the

Copernicus Sentinel-1/2 missions; originated from research code in

[DamageMappingTerramind](https://github.com/JulinaM/DamageMappingTerramind).

 

## License

 

MIT — see [`LICENSE`](LICENSE).

