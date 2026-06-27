#!/usr/bin/env python
"""Run or print unified external baseline commands.

This script intentionally keeps third-party repositories independent. It
generates a small RetailCabinet-4 override config for MMAction2, then delegates
training/testing to the official project entrypoints.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXTERNAL_ROOT = SCRIPT_DIR.parent
EXPERIMENT_FILE = EXTERNAL_ROOT / "experiments" / "baselines.yaml"


def load_experiment_file(path: Path = EXPERIMENT_FILE) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not JSON-compatible and PyYAML is not installed."
            ) from exc
        return yaml.safe_load(text)


def model_by_id(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model in config["models"]:
        if model["model_id"] == model_id:
            return model
    known = ", ".join(model["model_id"] for model in config["models"])
    raise KeyError(f"Unknown model_id={model_id!r}. Known models: {known}")


def find_first_glob(root: Path, patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root.glob(pattern)))
    if not matches:
        joined = "\n  ".join(patterns)
        raise FileNotFoundError(f"No base config found under {root} for:\n  {joined}")
    return matches[0].resolve()


def resolve_data_root(defaults: dict[str, Any]) -> Path:
    configured_root = defaults.get("data_root")
    if configured_root:
        root = Path(configured_root)
        if not root.is_absolute():
            root = EXTERNAL_ROOT / root
        return root.resolve()

    ann_dir = (EXTERNAL_ROOT / defaults["annotation_dir"]).resolve()
    metadata_file = ann_dir / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata_root = metadata.get("data_root")
        if metadata_root:
            root = Path(metadata_root)
            if not root.is_absolute():
                root = EXTERNAL_ROOT / root
            return root.resolve()

    raise RuntimeError("No data_root found in baselines.yaml defaults or metadata.json.")


def resolve_run_namespace(defaults: dict[str, Any]) -> str:
    namespace = str(defaults.get("run_namespace", "")).strip().strip("/\\")
    if any(part == ".." for part in Path(namespace).parts):
        raise ValueError(f"Invalid run_namespace={namespace!r}")
    return namespace


def resolve_eval_num_crops(model: dict[str, Any], override: int | None) -> int:
    num_crops = int(override if override is not None else model.get("test_num_crops", 3))
    if num_crops not in (1, 3):
        raise ValueError(
            f"--num-crop supports 1 or 3 for the current evaluation pipelines, "
            f"but got {num_crops}."
        )
    return num_crops


def resolve_class_balanced_loss(
    model: dict[str, Any],
    defaults: dict[str, Any],
    override: bool | None,
) -> bool:
    if override is not None:
        return override
    return bool(model.get("class_balanced_loss", defaults.get("class_balanced_loss", False)))


def compute_class_weights(ann_file: Path, num_classes: int) -> tuple[list[int], list[float]]:
    counts = [0 for _ in range(num_classes)]
    for line_number, line in enumerate(ann_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            label = int(line.split()[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid label in {ann_file}:{line_number}: {line!r}") from exc
        if label < 0 or label >= num_classes:
            raise ValueError(
                f"Label {label} in {ann_file}:{line_number} is outside [0, {num_classes})."
            )
        counts[label] += 1

    missing = [str(index) for index, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(
            f"Cannot build class-balanced CE weights because class(es) "
            f"{', '.join(missing)} have zero training samples in {ann_file}."
        )

    total = sum(counts)
    weights = [total / (num_classes * count) for count in counts]
    return counts, weights


def python_float_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.10g}" for value in values) + "]"


def write_mmaction_config(
    model: dict[str, Any],
    defaults: dict[str, Any],
    seed: int,
    repo_dir: Path,
    eval_split: str = "test",
    config_role: str = "train",
    eval_num_crops: int | None = None,
    class_balanced_loss: bool = False,
) -> Path:
    base_config = find_first_glob(repo_dir, model.get("base_config_globs", []))
    ann_dir = (EXTERNAL_ROOT / defaults["annotation_dir"]).resolve()
    data_root = resolve_data_root(defaults)
    frames = int(model.get("frames", 16))
    input_size = int(model.get("input_size", 224))
    resize_short = int(model.get("resize_short", max(input_size, round(input_size * 256 / 224))))
    test_num_clips = int(model.get("test_num_clips", 1))
    test_num_crops = resolve_eval_num_crops(model, eval_num_crops)
    report_resize_scale = "(-1, resize_short)" if test_num_crops == 1 else "(-1, input_size)"
    report_crop = (
        "dict(type='CenterCrop', crop_size=input_size)"
        if test_num_crops == 1
        else "dict(type='ThreeCrop', crop_size=input_size)"
    )
    train_batch_size = int(model.get("train_batch_size", 2))
    test_batch_size = int(model.get("test_batch_size", 1))
    train_epochs = int(model.get("train_epochs", defaults.get("train_epochs", 30)))
    warmup_epochs = int(model.get("warmup_epochs", defaults.get("warmup_epochs", 5)))
    optimizer_lr = float(model.get("optimizer_lr", defaults.get("optimizer_lr", 2.2e-5)))
    optimizer_lr_min = float(model.get("optimizer_lr_min", defaults.get("optimizer_lr_min", optimizer_lr / 100)))
    optimizer_weight_decay = float(
        model.get("optimizer_weight_decay", defaults.get("optimizer_weight_decay", 0.001))
    )
    optimizer_betas = tuple(
        float(x)
        for x in model.get("optimizer_betas", defaults.get("optimizer_betas", [0.9, 0.98]))
    )
    optimizer_eps = float(model.get("optimizer_eps", defaults.get("optimizer_eps", 1e-8)))
    warmup_start_factor = float(
        model.get("warmup_start_factor", defaults.get("warmup_start_factor", 1e-6))
    )
    optim_wrapper_type = str(
        model.get("mmaction_optim_wrapper_type", defaults.get("mmaction_optim_wrapper_type", "AmpOptimWrapper"))
    )
    target_effective_batch_size = int(
        model.get(
            "target_effective_train_batch_size",
            defaults.get("target_effective_train_batch_size", train_batch_size),
        )
    )
    train_accumulation_steps = int(
        model.get(
            "train_accumulation_steps",
            max(1, target_effective_batch_size // train_batch_size),
        )
    )
    checkpoint_interval = int(model.get("checkpoint_interval", defaults.get("checkpoint_interval", 1)))
    checkpoint_save_best = model.get(
        "checkpoint_save_best",
        defaults.get("checkpoint_save_best", "auto"),
    )
    checkpoint_save_last = bool(
        model.get("checkpoint_save_last", defaults.get("checkpoint_save_last", True))
    )
    checkpoint_max_keep_ckpts = int(
        model.get("checkpoint_max_keep_ckpts", defaults.get("checkpoint_max_keep_ckpts", 3))
    )
    namespace = resolve_run_namespace(defaults)
    generated_dir = EXTERNAL_ROOT / "configs" / "generated"
    if namespace:
        generated_dir = generated_dir / namespace
    generated_dir.mkdir(parents=True, exist_ok=True)
    out_file = generated_dir / f"{model['model_id']}_seed{seed}_{config_role}.py"

    num_classes = int(defaults.get("num_classes", 4))
    class_counts: list[int] = []
    class_weights: list[float] = []
    if class_balanced_loss:
        class_counts, class_weights = compute_class_weights(ann_dir / "train.txt", num_classes)
    class_weights_literal = python_float_list(class_weights)

    loss_cls_override = (
        f"loss_cls=dict(type='CrossEntropyLoss', class_weight={class_weights_literal})"
        if class_balanced_loss
        else "loss_cls=dict(type='CrossEntropyLoss')"
    )
    cls_head_override = (
        "dict("
        "num_classes=num_classes, "
        f"{loss_cls_override})"
    )
    data_preprocessor_override = (
        "dict("
        "type='ActionDataPreprocessor', "
        "mean=[123.675, 116.28, 103.53], "
        "std=[58.395, 57.12, 57.375], "
        "format_shape='NCTHW')"
    )
    if model.get("override_backbone_num_frames", False):
        model_override = (
            "model = dict("
            "backbone=dict(num_frames=clip_len), "
            f"cls_head={cls_head_override}, "
            f"data_preprocessor={data_preprocessor_override})"
        )
    else:
        model_override = (
            "model = dict("
            f"cls_head={cls_head_override}, "
            f"data_preprocessor={data_preprocessor_override})"
        )

    # MMAction2 deep-merges dictionaries from _base_. We override dataset
    # locations, class count, seed, sampling, evaluation, and optional
    # frame-count-dependent backbone fields while official modules remain intact.
    content = f"""# Auto-generated by external_baselines/scripts/run_experiment.py
_base_ = r'{base_config.as_posix()}'
custom_imports = dict(
    imports=['mmaction_retail_metrics', 'mmaction_retail_transforms', 'mmaction_retail_optim'],
    allow_failed_imports=False)

dataset_type = 'VideoDataset'
data_root = r'{data_root.as_posix()}'
ann_dir = r'{ann_dir.as_posix()}'
num_classes = {num_classes}
clip_len = {frames}
input_size = {input_size}
resize_short = {resize_short}
test_num_clips = {test_num_clips}
test_num_crops = {test_num_crops}
train_accumulation_steps = {train_accumulation_steps}
target_effective_train_batch_size = {train_batch_size * train_accumulation_steps}
train_epochs = {train_epochs}
warmup_epochs = {warmup_epochs}
base_lr = {optimizer_lr}
min_lr = {optimizer_lr_min}
optimizer_weight_decay = {optimizer_weight_decay}
optimizer_betas = {optimizer_betas!r}
optimizer_eps = {optimizer_eps}
warmup_start_factor = {warmup_start_factor}
optim_wrapper_type = '{optim_wrapper_type}'
report_eval_split = '{eval_split}'
class_balanced_loss = {class_balanced_loss}
class_counts = {class_counts!r}
class_weights = {class_weights_literal}
checkpoint_interval = {checkpoint_interval}
checkpoint_save_best = {checkpoint_save_best!r}
checkpoint_save_last = {checkpoint_save_last}
checkpoint_max_keep_ckpts = {checkpoint_max_keep_ckpts}

{model_override}

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, resize_short)),
    dict(
        type='MultiScaleCrop',
        input_size=input_size,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='RetailColorJitter',
        p=0.8,
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1),
    dict(type='RetailGrayScale', p=0.2),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='UniformSample',
        clip_len=clip_len,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, resize_short)),
    dict(type='CenterCrop', crop_size=input_size),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='UniformSample',
        clip_len=clip_len,
        num_clips=test_num_clips,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale={report_resize_scale}),
    {report_crop},
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size={train_batch_size},
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_dir + '/train.txt',
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size={test_batch_size},
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_dir + '/val.txt',
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size={test_batch_size},
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_dir + '/' + report_eval_split + '.txt',
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='RetailMetric', num_clips=1, num_crops=1)
test_evaluator = dict(type='RetailMetric', num_clips=test_num_clips, num_crops=test_num_crops)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=train_epochs, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type=optim_wrapper_type,
    optimizer=dict(
        type='RetailAdamW',
        lr=base_lr,
        weight_decay=optimizer_weight_decay,
        betas=optimizer_betas,
        eps=optimizer_eps),
    accumulative_counts=train_accumulation_steps,
    clip_grad=dict(max_norm=40, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=warmup_start_factor,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max(1, train_epochs - warmup_epochs),
        eta_min=min_lr,
        by_epoch=True,
        begin=warmup_epochs,
        end=train_epochs)
]
default_hooks = dict(
    checkpoint=dict(
        interval=checkpoint_interval,
        save_best=checkpoint_save_best,
        save_last=checkpoint_save_last,
        max_keep_ckpts=checkpoint_max_keep_ckpts))
randomness = dict(seed={seed}, deterministic=False)
auto_scale_lr = dict(enable=False)
"""
    out_file.write_text(content, encoding="utf-8")
    return out_file.resolve()


def write_vifi_config(
    model: dict[str, Any],
    defaults: dict[str, Any],
    seed: int,
    eval_split: str = "val",
    config_role: str = "train",
    eval_num_crops: int = 1,
    class_balanced_loss: bool = False,
) -> Path:
    ann_dir = (EXTERNAL_ROOT / defaults["annotation_dir"]).resolve()
    data_root = resolve_data_root(defaults)
    namespace = resolve_run_namespace(defaults)
    generated_dir = EXTERNAL_ROOT / "configs" / "generated"
    if namespace:
        generated_dir = generated_dir / namespace
    generated_dir.mkdir(parents=True, exist_ok=True)
    out_file = generated_dir / f"{model['model_id']}_seed{seed}_{config_role}.yaml"
    frames = int(model.get("frames", 16))
    train_batch_size = int(model.get("train_batch_size", 4))
    target_effective_batch_size = int(
        model.get(
            "target_effective_train_batch_size",
            defaults.get("target_effective_train_batch_size", train_batch_size),
        )
    )
    train_accumulation_steps = int(
        model.get(
            "train_accumulation_steps",
            max(1, target_effective_batch_size // train_batch_size),
        )
    )
    val_file = ann_dir / f"{eval_split}.txt"
    num_classes = int(defaults.get('num_classes', 4))
    class_counts: list[int] = []
    class_weights: list[float] = []
    if class_balanced_loss:
        class_counts, class_weights = compute_class_weights(ann_dir / "train.txt", num_classes)
    class_weights_literal = python_float_list(class_weights)

    content = f"""BASE: ['']
DATA:
  ROOT: '{data_root.as_posix()}'
  TRAIN_FILE: '{(ann_dir / 'train.txt').as_posix()}'
  VAL_FILE: '{val_file.as_posix()}'
  DATASET: {defaults.get('dataset', 'retail4_cjj')}
  INPUT_SIZE: 224
  NUM_FRAMES: {frames}
  NUM_CLASSES: {num_classes}
  LABEL_LIST: '{(ann_dir / 'labels.csv').as_posix()}'
  TRAIN_CROP_TYPE: MultiScaleCrop
  TRAIN_CROP_SCALES: [1.0, 0.875, 0.75, 0.66]
  TRAIN_RANDOM_CROP: False
  TRAIN_MAX_WH_SCALE_GAP: 1
MODEL:
  ARCH: ViT-B/16
TRAIN:
  EPOCHS: 30
  WARMUP_EPOCHS: 5
  BATCH_SIZE: {train_batch_size}
  ACCUMULATION_STEPS: {train_accumulation_steps}
  LR: 2.2e-05
  OPT_LEVEL: O0
AUG:
  LABEL_SMOOTH: 0.1
  COLOR_JITTER: 0.8
  GRAY_SCALE: 0.2
  MIXUP: 0.8
  CUTMIX: 1.0
  MIXUP_SWITCH_PROB: 0.5
LOSS:
  CLASS_BALANCED: {str(class_balanced_loss)}
  CLASS_COUNTS: {class_counts}
  CLASS_WEIGHTS: {class_weights_literal}
TEST:
  MULTI_VIEW_INFERENCE: {str(eval_num_crops != 1)}
  NUM_CLIP: 1
  NUM_CROP: {eval_num_crops}
TRAINER:
  ViFi_CLIP:
    ZS_EVAL: False
    USE: both
SEED: {seed}
"""
    out_file.write_text(content, encoding="utf-8")
    return out_file.resolve()


def find_checkpoint(work_dir: Path, configured: str | None) -> Path:
    if configured:
        checkpoint = Path(configured)
        if not checkpoint.is_absolute():
            checkpoint = EXTERNAL_ROOT / checkpoint
        return checkpoint.resolve()
    candidates = sorted(work_dir.rglob("best*.pth"))
    if not candidates:
        candidates = sorted(work_dir.rglob("epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {work_dir}. Pass --checkpoint or train first."
        )
    return candidates[-1].resolve()


def build_run_log_path(
    model_id: str,
    seed: int,
    stage: str,
    num_clip: int,
    num_crop: int,
    namespace: str = "",
) -> Path:
    log_root = EXTERNAL_ROOT / "run_logs"
    if namespace:
        log_root = log_root / namespace
    log_dir = (log_root / model_id / f"seed{seed}").resolve()
    if stage in {"val", "test"}:
        log_name = f"{stage}_clip{num_clip}_crop{num_crop}.log"
    else:
        log_name = f"{stage}.log"
    return log_dir / log_name


def build_context(
    config: dict[str, Any],
    model: dict[str, Any],
    seed: int,
    stage: str,
    checkpoint_override: str | None,
    num_crop_override: int | None,
    class_balanced_override: bool | None,
) -> dict[str, str]:
    defaults = config["defaults"]
    repo = model.get("repo", "")
    repo_dir = (EXTERNAL_ROOT / repo).resolve() if repo else EXTERNAL_ROOT
    eval_split = "val" if stage == "val" else "test"
    config_role = stage if stage in {"val", "test"} else "train"
    if stage in {"val", "test"}:
        eval_num_crops = resolve_eval_num_crops(model, num_crop_override)
    else:
        eval_num_crops = resolve_eval_num_crops(
            model, num_crop_override if num_crop_override is not None else 1)
    eval_num_clips = int(model.get("test_num_clips", 1))
    multi_view_inference = eval_num_clips * eval_num_crops > 1
    class_balanced_loss = resolve_class_balanced_loss(model, defaults, class_balanced_override)
    class_counts: list[int] = []
    class_weights: list[float] = []
    if class_balanced_loss:
        class_counts, class_weights = compute_class_weights(
            (EXTERNAL_ROOT / defaults["annotation_dir"] / "train.txt").resolve(),
            int(defaults.get("num_classes", 4)),
        )
    namespace = resolve_run_namespace(defaults)
    work_dir_root = EXTERNAL_ROOT / defaults.get("work_dir_root", "work_dirs")
    if namespace:
        work_dir_root = work_dir_root / namespace
    work_dir = (work_dir_root / model["model_id"] / f"seed{seed}").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    model_config = model.get("config") or ""
    if model["backend"] == "mmaction2" and (stage != "print" or repo_dir.exists()):
        model_config = write_mmaction_config(
            model,
            defaults,
            seed,
            repo_dir,
            eval_split,
            config_role,
            eval_num_crops,
            class_balanced_loss,
        ).as_posix()
    elif model["backend"] == "mmaction2":
        patterns = model.get("base_config_globs", [])
        model_config = patterns[0] if patterns else ""
    elif model["backend"] == "vifi_clip":
        vifi_eval_split = eval_split if stage in {"val", "test"} else "val"
        model_config = write_vifi_config(
            model,
            defaults,
            seed,
            vifi_eval_split,
            config_role,
            eval_num_crops,
            class_balanced_loss,
        ).as_posix()

    checkpoint = ""
    if stage in {"test", "val"}:
        checkpoint = find_checkpoint(work_dir, checkpoint_override or model.get("checkpoint")).as_posix()
    elif checkpoint_override:
        checkpoint = str(Path(checkpoint_override).resolve())
    dump_file = (work_dir / f"{stage}_outputs.pkl").resolve()
    run_log_file = build_run_log_path(
        model["model_id"],
        seed,
        stage,
        eval_num_clips,
        eval_num_crops,
        namespace,
    )

    return {
        "root": EXTERNAL_ROOT.as_posix(),
        "external_root": EXTERNAL_ROOT.as_posix(),
        "repo": repo_dir.as_posix(),
        "data_root": resolve_data_root(defaults).as_posix(),
        "ann_dir": str((EXTERNAL_ROOT / defaults["annotation_dir"]).resolve()).replace("\\", "/"),
        "work_dir": work_dir.as_posix(),
        "model_id": model["model_id"],
        "seed": str(seed),
        "frames": str(model.get("frames", "")),
        "config": str(model_config).replace("\\", "/"),
        "checkpoint": checkpoint,
        "eval_split": eval_split,
        "num_clip": str(eval_num_clips),
        "num_crop": str(eval_num_crops),
        "multi_view_inference": str(multi_view_inference),
        "class_balanced_loss": str(class_balanced_loss),
        "class_counts": ",".join(str(count) for count in class_counts),
        "class_weights": ",".join(f"{weight:.10g}" for weight in class_weights),
        "dump_file": dump_file.as_posix(),
        "run_log_file": run_log_file.as_posix(),
        "run_log_dir": run_log_file.parent.as_posix(),
        "python": sys.executable.replace("\\", "/"),
    }


def command_for_stage(model: dict[str, Any], stage: str) -> str:
    if stage == "train":
        key = "train_cmd"
    elif stage == "smoke":
        key = "smoke_cmd"
    elif stage in {"test", "val"}:
        key = "eval_cmd"
    else:
        raise ValueError(stage)
    command = model.get(key, "")
    if not command:
        raise RuntimeError(f"Model {model['model_id']} does not define {key}")
    return command


def run_command(
    command: str,
    cwd: Path,
    dry_run: bool,
    log_file: Path,
    metadata: dict[str, str],
) -> int:
    print(f"[cwd] {cwd}")
    print(f"[cmd] {command}")
    print(f"[log] {log_file}")
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    stub_path = (EXTERNAL_ROOT / "stubs").resolve().as_posix()
    custom_path = EXTERNAL_ROOT.resolve().as_posix()
    existing_pythonpath = env.get("PYTHONPATH", "")
    extra_pythonpath = os.pathsep.join([stub_path, custom_path])
    env["PYTHONPATH"] = (
        extra_pythonpath if not existing_pythonpath
        else extra_pythonpath + os.pathsep + existing_pythonpath
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write("# external_baselines stage log\n")
        handle.write(f"started_at: {started_at}\n")
        handle.write(f"cwd: {cwd}\n")
        handle.write(f"cmd: {command}\n")
        for key in (
            "model_id",
            "seed",
            "stage",
            "frames",
            "eval_split",
            "num_clip",
            "num_crop",
            "multi_view_inference",
            "class_balanced_loss",
            "class_counts",
            "class_weights",
            "config",
            "checkpoint",
            "work_dir",
            "dump_file",
        ):
            value = metadata.get(key, "")
            handle.write(f"{key}: {value}\n")
        handle.write("\n")
        handle.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        returncode = process.wait()
        ended_at = datetime.now().isoformat(timespec="seconds")
        handle.write("\n")
        handle.write(f"ended_at: {ended_at}\n")
        handle.write(f"exit_code: {returncode}\n")
    print(f"[exit_code] {returncode}")
    print(f"[log] {log_file}")
    return returncode


def list_models(config: dict[str, Any]) -> None:
    for model in config["models"]:
        print(
            f"{model['model_id']}\t{model.get('backend', '')}\t"
            f"{model.get('frames', '')}\t{model.get('display_name', '')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-file",
        default=str(EXPERIMENT_FILE),
        help="Experiment registry to use. Defaults to experiments/baselines.yaml.",
    )
    parser.add_argument("--model", help="Model id from experiments/baselines.yaml")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--stage", choices=["train", "smoke", "test", "val", "print"], default="print")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint for test stage.")
    parser.add_argument(
        "--num-crop",
        type=int,
        choices=[1, 3],
        default=None,
        help="For standalone val/test: 1 uses CenterCrop single-view; 3 uses ThreeCrop multi-view. Defaults to model test_num_crops.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    class_balance_group = parser.add_mutually_exclusive_group()
    class_balance_group.add_argument(
        "--class-balanced-loss",
        action="store_true",
        default=None,
        help="Enable inverse-frequency class weights in the training cross-entropy loss.",
    )
    class_balance_group.add_argument(
        "--no-class-balanced-loss",
        action="store_false",
        dest="class_balanced_loss",
        help="Disable class-balanced cross-entropy even if enabled in baselines.yaml.",
    )
    args = parser.parse_args()

    config = load_experiment_file(Path(args.experiment_file))
    if args.list:
        list_models(config)
        return 0
    if not args.model:
        parser.error("--model is required unless --list is used")

    model = model_by_id(config, args.model)
    if model.get("backend") == "tc_clip_existing":
        print("Existing TC-CLIP rows are parsed by collect_results.py; no command to run.")
        return 0

    context = build_context(
        config,
        model,
        args.seed,
        args.stage,
        args.checkpoint,
        args.num_crop,
        args.class_balanced_loss,
    )
    if args.stage == "print":
        print(json.dumps(context, indent=2, ensure_ascii=False))
        return 0

    command = command_for_stage(model, args.stage).format(**context)
    repo_dir = Path(context["repo"])
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Repository directory does not exist: {repo_dir}. "
            "Run scripts/download_repos.py first."
        )
    log_metadata = dict(context)
    log_metadata["stage"] = args.stage
    return run_command(
        command,
        repo_dir,
        args.dry_run,
        Path(context["run_log_file"]),
        log_metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
