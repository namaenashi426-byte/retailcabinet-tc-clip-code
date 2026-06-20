"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/main.py
"""

import wandb
from apex import amp
import time
import torch
import torch.distributed as dist

from utils.tools import accuracy_top1_top5
from utils.logger import MetricLogger, SmoothedValue


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, logger, config, mixup_fn):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = 'Epoch: [{}]'.format(epoch)

    for idx, batch_data in enumerate(metric_logger.log_every(train_loader, config.print_freq, logger, header)):
        images = batch_data['imgs'].cuda(non_blocking=True)
        label_id = batch_data['label'].cuda(non_blocking=True)
        label_id = label_id.reshape(-1) # [b]
        images = images.view((-1, config.num_frames, 3) + images.size()[-2:])  # [b, t, c, h, w]

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)   # label_id [b] -> [b, num_class]

        # forward
        output = model(images)
        total_loss = criterion(output["logits"], label_id)
        total_loss_divided = total_loss / config.accumulation_steps

        # backward
        if config.accumulation_steps == 1:
            optimizer.zero_grad()
        if config.opt_level != 'O0':
            with amp.scale_loss(total_loss_divided, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss_divided.backward()
        if config.accumulation_steps > 1:
            if (idx + 1) % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        log_stats = metric_logger.get_stats(prefix='train_inner/')
        if dist.get_rank() == 0 and config.use_wandb:
            wandb.log(log_stats, step=epoch*num_steps+idx)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.get_stats()


@torch.no_grad()
def validate(val_loader, model, logger, config):
    model.eval()
    num_classes = len(val_loader.dataset.classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long).cuda()
    total_clips = torch.zeros(1, dtype=torch.long).cuda()

    logger.info(f"{config.num_clip * config.num_crop} views inference")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for idx, batch_data in enumerate(metric_logger.log_every(val_loader, config.print_freq, logger, header)):
        _image = batch_data["imgs"]  # [b, tn, c, h, w]
        label_id = batch_data["label"]
        label_id = label_id.reshape(-1)  # [b]

        b, tn, c, h, w = _image.size()
        t = config.num_frames
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)

        tot_similarity = torch.zeros((b, num_classes)).cuda()
        for i in range(n):
            image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
            label_id = label_id.cuda(non_blocking=True)
            image_input = image.cuda(non_blocking=True)

            if config.opt_level == 'O2':
                image_input = image_input.half()

            output = model(image_input)
            logits = output["logits"]
            similarity = logits.view(b, -1).softmax(dim=-1)
            tot_similarity += similarity

        # Classification score
        acc1, acc5, indices_1, _ = accuracy_top1_top5(tot_similarity, label_id)
        metric_logger.meters['acc1'].update(float(acc1) / b * 100, n=b)
        metric_logger.meters['acc5'].update(float(acc5) / b * 100, n=b)
        pred = tot_similarity.argmax(dim=-1)
        label_cuda = label_id.cuda(non_blocking=True)
        valid = (label_cuda >= 0) & (label_cuda < num_classes)
        if valid.any():
            encoded = label_cuda[valid] * num_classes + pred[valid]
            confusion += torch.bincount(
                encoded.long(), minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
        total_clips += b * n

    metric_logger.synchronize_between_processes()
    dist.all_reduce(confusion)
    dist.all_reduce(total_clips)
    elapsed = max(time.time() - start_time, 1e-6)

    confusion_float = confusion.float()
    support = confusion_float.sum(dim=1)
    pred_count = confusion_float.sum(dim=0)
    correct = torch.diag(confusion_float)
    valid_class = support > 0
    recall = torch.zeros_like(support)
    precision = torch.zeros_like(pred_count)
    recall[valid_class] = correct[valid_class] / support[valid_class]
    precision[pred_count > 0] = correct[pred_count > 0] / pred_count[pred_count > 0]
    f1 = torch.zeros_like(support)
    denom = precision + recall
    f1[denom > 0] = 2 * precision[denom > 0] * recall[denom > 0] / denom[denom > 0]
    balanced_acc = recall[valid_class].mean().item() * 100 if valid_class.any() else 0.0
    macro_f1 = f1[valid_class].mean().item() * 100 if valid_class.any() else 0.0
    clips_per_second = float(total_clips.item()) / elapsed
    ms_per_clip = 1000.0 / max(clips_per_second, 1e-12)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    logger.info(f' * Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    logger.info(f' * Macro-F1 {macro_f1:.3f} Balanced Acc {balanced_acc:.3f} '
                f'Clips/s {clips_per_second:.2f} ms/clip {ms_per_clip:.2f} '
                f'Peak Mem {peak_mem_mb:.1f} MB')
    logger.info(f' * Per-class recall {recall.detach().cpu().numpy().round(4).tolist()}')
    logger.info(f' * Confusion matrix {confusion.detach().cpu().numpy().astype(int).tolist()}')

    stats = metric_logger.get_stats()
    stats.update({
        'macro_f1': macro_f1,
        'balanced_acc': balanced_acc,
        'clips_per_second': clips_per_second,
        'ms_per_clip': ms_per_clip,
        'peak_mem_mb': peak_mem_mb,
    })
    return stats
