"""Minimal Apex AMP stub for repositories that import apex unconditionally.

This is only intended for OPT_LEVEL=O0 runs where real mixed precision is not
used. It keeps ViFi-CLIP importable on Windows environments without NVIDIA Apex.
"""

from __future__ import annotations

from contextlib import contextmanager


class _AmpStub:
    def initialize(self, models=None, optimizers=None, opt_level="O0", **kwargs):
        if models is None and optimizers is None:
            return None
        if optimizers is None:
            return models
        if models is None:
            return optimizers
        return models, optimizers

    @contextmanager
    def scale_loss(self, loss, optimizer):
        yield loss


amp = _AmpStub()
