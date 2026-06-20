"""RetailCabinet-4 metrics for MMAction2 external baselines."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmaction.registry import METRICS


@METRICS.register_module()
class RetailMetric(BaseMetric):
    """Report metrics aligned with the main TC-CLIP experiments."""

    default_prefix: Optional[str] = "acc"

    def __init__(
        self,
        num_clips: int = 1,
        num_crops: int = 1,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_clips = int(num_clips)
        self.num_crops = int(num_crops)
        self._start_time: float | None = None

    def process(self, data_batch, data_samples: Sequence[Dict]) -> None:
        if self._start_time is None:
            self._start_time = time.time()
        for data_sample in data_samples:
            pred = data_sample["pred_score"].detach().cpu().numpy()
            label = data_sample["gt_label"]
            if label.size(0) == 1:
                label = int(label.item())
            else:
                label = label.detach().cpu().numpy()
            self.results.append({"pred": pred, "label": label})

    def compute_metrics(self, results: List[Dict]) -> Dict:
        if not results:
            return OrderedDict()

        scores = np.stack([item["pred"] for item in results])
        labels = np.asarray([item["label"] for item in results], dtype=np.int64)
        num_classes = scores.shape[1]
        top1_pred = scores.argmax(axis=1)
        topk = min(5, num_classes)
        topk_pred = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]

        top1 = float((top1_pred == labels).mean() * 100.0)
        top5 = float(np.asarray([
            labels[i] in topk_pred[i] for i in range(labels.shape[0])
        ]).mean() * 100.0)

        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        valid = (labels >= 0) & (labels < num_classes)
        for gt, pred in zip(labels[valid], top1_pred[valid]):
            confusion[int(gt), int(pred)] += 1

        support = confusion.sum(axis=1).astype(np.float64)
        pred_count = confusion.sum(axis=0).astype(np.float64)
        correct = np.diag(confusion).astype(np.float64)
        valid_class = support > 0

        recall = np.zeros(num_classes, dtype=np.float64)
        precision = np.zeros(num_classes, dtype=np.float64)
        recall[valid_class] = correct[valid_class] / support[valid_class]
        nonzero_pred = pred_count > 0
        precision[nonzero_pred] = correct[nonzero_pred] / pred_count[nonzero_pred]

        f1 = np.zeros(num_classes, dtype=np.float64)
        denom = precision + recall
        valid_f1 = denom > 0
        f1[valid_f1] = 2 * precision[valid_f1] * recall[valid_f1] / denom[valid_f1]

        macro_f1 = float(f1[valid_class].mean() * 100.0) if valid_class.any() else 0.0
        balanced_acc = float(recall[valid_class].mean() * 100.0) if valid_class.any() else 0.0
        elapsed = max(time.time() - self._start_time, 1e-6) if self._start_time else 1e-6
        total_clips = labels.shape[0] * self.num_clips * self.num_crops
        clips_s = float(total_clips / elapsed)
        ms_clip = float(1000.0 / max(clips_s, 1e-12))
        peak_mem_mb = (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        )

        logger = MMLogger.get_current_instance()
        logger.info(f" * Acc@1 {top1:.3f} Acc@5 {top5:.3f}")
        logger.info(
            f" * Macro-F1 {macro_f1:.3f} Balanced Acc {balanced_acc:.3f} "
            f"Clips/s {clips_s:.2f} ms/clip {ms_clip:.2f} "
            f"Peak Mem {peak_mem_mb:.1f} MB"
        )
        logger.info(f" * Per-class recall {np.round(recall, 4).tolist()}")
        logger.info(f" * Confusion matrix {confusion.astype(int).tolist()}")
        self._start_time = None

        return OrderedDict([
            ("top1", top1),
            ("top5", top5),
            ("macro_f1", macro_f1),
            ("bacc", balanced_acc),
            ("clips_s", clips_s),
            ("ms_clip", ms_clip),
            ("peak_mem_mb", peak_mem_mb),
        ])
