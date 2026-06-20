"""Optimizer adapters for RetailCabinet MMAction2 baselines."""

from __future__ import annotations

from torch.optim import AdamW

from mmengine.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class RetailAdamW(AdamW):
    """AdamW that tolerates optimizer keys inherited from MMAction2 SGD bases."""

    def __init__(self, params, **kwargs):
        for key in ("_delete_", "momentum", "dampening", "nesterov"):
            kwargs.pop(key, None)
        super().__init__(params, **kwargs)
