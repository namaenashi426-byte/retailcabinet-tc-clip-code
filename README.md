# RetailCabinet TC-CLIP

This repository contains the code used for the paper:

**A Fine-Grained Action Recognition Method for Smart Retail Cabinets Combining Motion-Aware Sampling and Temporal Difference Modeling**

This release is built on the published
[TC-CLIP](https://github.com/naver-ai/tc-clip) codebase and includes:

- Motion-R / Motion-8 motion-aware frame sampling.
- Temporal Difference Head (TDH) for signed adjacent-frame feature differences.
- RetailCabinet-4 labels and fixed train/validation/test split files.
- Upstream TC-CLIP SSv2 temporal split and label files reused for the
  SSV2-Temporal18 generalization checks.
- Tabular source data used to generate the reported figures.
- External-baseline orchestration code for MMAction2 and ViFi-CLIP comparisons.

## Data Availability

The raw RetailCabinet-4 videos are not included in this repository. They contain human-participant recordings and are subject to privacy, consent, and institutional access restrictions.

The included files support reproducible setup and result inspection:

- `labels/retail4_cjj_labels.csv`: class names.
- `datasets_splits/`: RetailCabinet-4 splits and retained upstream TC-CLIP
  split resources used by this release.
- `labels/labels_temporal_static/ssv2_temporal18_labels.csv` and
  `datasets_splits/temporal_static_splits/ssv2_splits/`: upstream TC-CLIP
  SSV2-Temporal18 label and split files reused for public-dataset
  generalization experiments.
- `source_data/`: processed tabular source data for the paper figures.
- `configs/data/fully_supervised_cjj.yaml`: RetailCabinet-4 data configuration.
- `external_baselines/`: comparison-experiment wrappers, generated config examples,
  RetailCabinet-4 and SSV2-Temporal18 annotation files, and aggregated baseline
  result tables.

To run training or evaluation, place the restricted RetailCabinet-4 video files in the structure referenced by the split files and set `retail4.root` in `configs/common/default.yaml` or override it on the command line.
For SSV2-Temporal18 experiments, obtain the public
[Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something)
videos and set `ssv2.root` locally. The SSv2 split and label files in this
release follow the upstream
[TC-CLIP](https://github.com/naver-ai/tc-clip) resources; they were not newly
constructed by this project.

Example structure:

```text
/path/to/RetailCabinet-4/videos/
  train/
    drink/
    open/
    purchase/
    putback/
  val/
    drink/
    open/
    purchase/
    putback/
  test/
    drink/
    open/
    purchase/
    putback/
```

## Installation

Create an environment following the original TC-CLIP requirements. A minimal starting point is:

```bash
pip install -r requirements.txt
```

The original code depends on PyTorch, CUDA, MMCV, Decord, Hydra, Apex AMP, and related video-recognition packages. See `docs/INSTALL.md` for the upstream installation notes.

## Training

The default paper setting uses Motion-8 and TDH on RetailCabinet-4:

```bash
bash scripts/train/train_retail4_motion8_tdh.sh /path/to/RetailCabinet-4/videos
```

The same command can be run manually:

```bash
torchrun --nproc_per_node=1 main.py -cn fully_supervised \
  data=fully_supervised_cjj \
  output=workspace/expr/retail4_motion8_tdh \
  trainer=tc_clip \
  use_wandb=false \
  retail4.root=/path/to/RetailCabinet-4/videos \
  sample_type=motion \
  motion_sampler.mode=motion \
  num_frames=8 \
  temporal_delta.enable=true \
  temporal_delta.mode=adjacent \
  temporal_delta.alpha=0.5 \
  temporal_delta.hidden_dim=256 \
  temporal_delta.dropout=0.2 \
  temporal_delta.detach_visual=true
```

## Evaluation

After training, evaluate a checkpoint with:

```bash
bash scripts/eval/eval_retail4_motion8_tdh.sh /path/to/RetailCabinet-4/videos /path/to/best.pth
```

## What Is Not Included

This release intentionally excludes:

- Raw RetailCabinet-4 video data.
- Model checkpoints and generated workspaces.
- Private experiment logs with machine-specific paths.
- Third-party baseline repositories downloaded under `external_baselines/repos/`.
- Manuscript drafts, submission documents, and local recovery files.

## License and Attribution

This project is based on [TC-CLIP](https://github.com/naver-ai/tc-clip) and retains the upstream license and notices in `LICENSE` and `NOTICE`. TC-CLIP is licensed under CC BY-NC 4.0. Subcomponents keep their original license terms as described in `NOTICE`.

## Citation

If you use this code, please cite the paper associated with this repository and the original TC-CLIP paper:

```bibtex
@inproceedings{kim2024tcclip,
  title={Leveraging Temporal Contextualization for Video Action Recognition},
  author={Kim, Minji and Han, Dongyoon and Kim, Taekyung and Han, Bohyung},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
