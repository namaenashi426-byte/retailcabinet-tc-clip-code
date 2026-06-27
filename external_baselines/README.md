# External Baselines for RetailCabinet-4

This folder is a unified orchestration layer for comparing TC-CLIP variants
with external video-understanding baselines. Third-party repositories stay in
`repos/`; this project only prepares annotations, generates local configs,
runs official entrypoints, and collects metrics.

This public release includes the project-owned orchestration code, generated
configuration examples, RetailCabinet-4 annotation files, and aggregated result
tables. It does not include third-party repositories, checkpoints, raw videos,
work directories, or private run logs. Paths in the committed generated configs
use placeholders such as `/path/to/RetailCabinet-4/videos`; regenerate configs
with `scripts/run_experiment.py` after setting `defaults.data_root` locally.

The release also includes SSV2-Temporal18 generalization material:
`experiments/baselines_ssv2_temporal18.yaml`,
`annotations/ssv2_temporal18/`, generated config examples under
`configs/generated/ssv2_temporal18/`, and
`results/ssv2_temporal18_generalization_summary.csv`. Raw Something-Something
V2 videos are not included.

## Layout

- `repos/`: cloned third-party repositories.
- `annotations/retail4/`: generated RetailCabinet-4 annotation files.
- `configs/generated/`: generated MMAction2 configs.
- `experiments/baselines.yaml`: model registry and command templates.
- `scripts/prepare_retail4.py`: convert and validate TC-CLIP split files.
- `scripts/download_repos.py`: clone external repositories.
- `scripts/run_experiment.py`: print, smoke-test, train, val, or test one model.
- `scripts/collect_results.py`: parse logs into `results/summary.csv`.
- `mmaction_retail_transforms.py`: MMAction2 transforms matching the main
  TC-CLIP color preprocessing.
- `work_dirs/`: external model outputs.
- `checkpoints/`: optional manually downloaded checkpoints.

## 1. Prepare Annotations

Run from this directory:

```powershell
python scripts/prepare_retail4.py --strict
```

Outputs include:

- `annotations/retail4/train.txt`
- `annotations/retail4/val.txt`
- `annotations/retail4/test.txt`
- `annotations/retail4/*_absolute.txt`
- `annotations/retail4/*.csv`
- `annotations/retail4/metadata.json`
- `annotations/retail4/split_stats.csv`

The label map is fixed:

- `0`: `drink`
- `1`: `open`
- `2`: `purchase`
- `3`: `putback`

## 2. Download Repositories

```powershell
python scripts/download_repos.py --repo required
```

Required repositories:

- `open-mmlab/mmaction2`
- `muzairkhattak/ViFi-CLIP`

Optional repositories can be cloned with:

```powershell
python scripts/download_repos.py --repo all
```

## 3. Create Isolated Environments

The current TC-CLIP shell may not contain `torch`, `mmcv`, or `decord`.
Use isolated conda environments so external baselines do not disturb the
working TC-CLIP environment.

MMAction2:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_mmaction2_env.ps1
```

ViFi-CLIP:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_vificlip_env.ps1
```

After installation, run commands through `conda run`, for example:

```powershell
conda run -n retail_mmaction2 python scripts/run_experiment.py --model videomae_v2_vitb_8f --seed 1024 --stage smoke
```

## 4. List Models

```powershell
python scripts/run_experiment.py --list
```

## 5. Print or Run a Command

Print the generated context:

```powershell
python scripts/run_experiment.py --model videomae_v2_vitb_8f --seed 1024 --stage print
```

Smoke test one epoch:

```powershell
python scripts/run_experiment.py --model videomae_v2_vitb_8f --seed 1024 --stage smoke
```

Train:

```powershell
python scripts/run_experiment.py --model videomae_v2_vitb_8f --seed 1024 --stage train
```

Report validation metrics:

```powershell
python scripts/run_experiment.py --model videomae_v2_vitb_8f --seed 1024 --stage val --num-crop 1
```

Final test evaluation:

```powershell
python scripts/run_experiment.py --model videomae_v2_vitb_8f --seed 1024 --stage test --num-crop 3
```

Enable class-balanced cross entropy for a run:

```powershell
python scripts/run_experiment.py --model videomae_v2_vitb_8f --seed 1024 --stage train --class-balanced-loss
```

Use `--dry-run` to print the exact command without running it.
For standalone `val` / `test`, `--num-crop 1` uses single-view `CenterCrop`;
`--num-crop 3` uses multi-view `ThreeCrop`. If omitted, the model registry
default is used, currently 3 crops.

For non-dry-run `train` / `smoke` / `val` / `test`, `run_experiment.py` writes
the latest stage log to a stable path under
`external_baselines/run_logs/<model_id>/seed<seed>/`:

- train: `train.log`
- val: `val_clip<num_clip>_crop<num_crop>.log`
- test: `test_clip<num_clip>_crop<num_crop>.log`

These stable logs are the primary logs for handoff and result collection. The
timestamped logs created inside `work_dirs/` are MMAction2/MMEngine internal
logs and can be treated as fallback/debug artifacts.
Stage log headers record whether class-balanced CE was enabled and the exact
class weights used.

When passing a relative `--checkpoint`, make it relative to this
`external_baselines/` folder, for example `--checkpoint checkpoints/model.pth`.

WSL one-shot MMAction2 run for the fair 8-frame MMAction2 grid:

```bash
cd /path/to/your/TC-CLIP-copy
bash external_baselines/scripts/run_mmaction2_8f_wsl.sh
```

The old `run_mmaction2_8f_3x3_wsl.sh` file is only a compatibility wrapper.

The script does not set a data path or regenerate annotations. It reads
`experiments/baselines.yaml` for the MMAction2 8-frame model ids, seeds, command
templates, batch sizes, and data root used by generated configs. The batch
script writes orchestration logs under `external_baselines/run_logs/batch/`,
while each invoked `run_experiment.py` call writes the stable per-stage logs
above. It pauses on exit by default and stops at the first failing stage. Set
`STOP_ON_ERROR=0` to keep going after failures or `PAUSE_ON_EXIT=0` to disable
the final prompt.
Set `EVAL_NUM_CROP=1` for single-view val/test in the WSL batch script; the
default is `EVAL_NUM_CROP=3`.

## 6. Collect Results

```powershell
python scripts/collect_results.py --out results/summary.csv
```

This writes:

- `results/summary.csv`: per-seed rows.
- `results/summary_agg.csv`: mean/std rows.

The parser also includes existing TC-CLIP logs from the main repository, so the
external baseline table can sit next to the current `Uniform-8`,
`Motion-only-8`, and `TemporalDeltaHead` results.
For external baseline rows, the parser first reads the stable
`run_logs/<model_id>/seed<seed>/test_clip*_crop*.log` and
`val_clip*_crop*.log` files, then falls back to `work_dirs/` logs.

For fair-budget comparisons with the main 8-frame TC-CLIP experiments, prefer
`videomae_v2_vitb_8f`, `video_swin_base_8f`, `slowfast_r101_8f`,
`timesformer_divst_8f`, and `vifi_clip_8f`. The native-frame entries remain in
the registry as supplemental comparisons.

Training preprocessing is aligned with the main TC-CLIP pipeline: full-video
uniform sampling, `Resize(short) -> MultiScaleCrop -> Resize(224) -> Flip ->
ColorJitter(p=0.8) -> GrayScale(p=0.2)`, then ImageNet mean/std
normalization. MMAction2 performs normalization in `ActionDataPreprocessor`;
ViFi-CLIP keeps it as an explicit pipeline transform.

Training uses gradient accumulation to match the main fully-supervised
effective batch size of 512. MMAction2 configs set
`optim_wrapper.accumulative_counts`; ViFi-CLIP configs set
`TRAIN.ACCUMULATION_STEPS`. With the current single-GPU commands,
VideoMAE-B/Swin-B/SlowFast-R101/TimeSformer-DivST use train batch 1 x accum
512, while ViFi-CLIP uses train batch 4 x accum 128. MMAction2 dataloaders use
`num_workers=4`; ViFi-CLIP uses `num_workers=16`.

MMAction2 training hyperparameters are also generated from `baselines.yaml`
defaults to match the main fully-supervised run: AdamW, lr `2.2e-5`, min lr
`2.2e-7`, weight decay `0.001`, betas `(0.9, 0.98)`, 30 epochs, 5 warmup
epochs, and `AmpOptimWrapper`.

Class-balanced CE is available but disabled by default in
`experiments/baselines.yaml` (`defaults.class_balanced_loss=false`). Enable it
with `--class-balanced-loss` for a single run, or set `class_balanced_loss` at
the default/model level in the registry. The weights are computed from
`annotations/retail4/train.txt` as `total / (num_classes * class_count)`;
current weights are `[1.9355, 2.2727, 0.4959, 0.9740]` for
`drink/open/purchase/putback`.

## Notes

- MMAction2 installation is environment-sensitive on Windows. Keep it in a
  dedicated conda environment. If `mmcv` installation fails, use WSL2 or Docker.
- MMAction2 model config file names may change across versions. The runner uses
  glob patterns from `experiments/baselines.yaml`; update those patterns if a
  cloned branch uses different config names.
- Pretrained checkpoints can be placed under `checkpoints/` and referenced in
  `experiments/baselines.yaml`.
