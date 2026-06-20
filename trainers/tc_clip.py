"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn

from trainers.tc_clip_text_encoder import VPTextEncoder
from trainers.tc_clip_prompt_learner import VPPromptLearner


class TCCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.prompt_learner = VPPromptLearner(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = VPTextEncoder(cfg, clip_model, logger)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype if cfg.opt_level != 'O0' else torch.float32
        self.prompt_generation_layer_level = self.text_encoder.prompt_generation_layer_level
        self.return_layer_num = self.prompt_generation_layer_level.copy()
        if 11 not in self.return_layer_num:
            self.return_layer_num.append(11)
        logger.info(f"Using context tokens from vision layer {self.return_layer_num}")

        self.temporal_delta_cfg = cfg.get("temporal_delta", {})
        self.use_temporal_delta = bool(self.temporal_delta_cfg.get("enable", False))
        if self.use_temporal_delta:
            feature_dim = self._infer_visual_feature_dim(clip_model)
            hidden_dim = int(self.temporal_delta_cfg.get("hidden_dim", 256))
            dropout = float(self.temporal_delta_cfg.get("dropout", 0.2))
            self.temporal_delta_alpha = float(self.temporal_delta_cfg.get("alpha", 0.5))
            self.temporal_delta_mode = str(
                self.temporal_delta_cfg.get("mode", "adjacent")).lower()
            valid_delta_modes = ["edge", "adjacent", "pair"]
            if self.temporal_delta_mode not in valid_delta_modes:
                raise ValueError(
                    f"temporal_delta.mode must be one of {valid_delta_modes}, "
                    f"got {self.temporal_delta_mode}"
                )
            self.temporal_delta_head = nn.Sequential(
                nn.LayerNorm(feature_dim * 4),
                nn.Linear(feature_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(classnames)),
            )
            logger.info(
                f"Using temporal delta head: mode={self.temporal_delta_mode}, "
                f"alpha={self.temporal_delta_alpha}, hidden_dim={hidden_dim}, "
                f"dropout={dropout}"
            )

    def _rebuild_classnames(self, cfg, classnames, clip_model, logger):
        self.prompt_learner._rebuild_classnames(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    @staticmethod
    def _infer_visual_feature_dim(clip_model):
        if hasattr(clip_model.visual, "output_dim"):
            return int(clip_model.visual.output_dim)
        return int(clip_model.text_projection.shape[1])

    def _edge_delta_feature(self, frame_features):
        edge_k = int(self.temporal_delta_cfg.get("edge_k", 2))
        edge_k = max(1, min(edge_k, frame_features.size(1)))
        early = frame_features[:, :edge_k, :].mean(dim=1)
        late = frame_features[:, -edge_k:, :].mean(dim=1)
        delta = late - early
        return torch.cat([early, late, delta, delta.abs()], dim=-1)

    def _adjacent_delta_feature(self, frame_features):
        if frame_features.size(1) <= 1:
            zeros = torch.zeros_like(frame_features[:, 0, :])
            return torch.cat([zeros, zeros, zeros, zeros], dim=-1)

        deltas = frame_features[:, 1:, :] - frame_features[:, :-1, :]
        mean_delta = deltas.mean(dim=1)
        first_delta = deltas[:, 0, :]
        last_delta = deltas[:, -1, :]
        mean_abs_delta = deltas.abs().mean(dim=1)
        return torch.cat(
            [mean_delta, first_delta, last_delta, mean_abs_delta], dim=-1)

    def _pair_delta_feature(self, frame_features):
        if frame_features.size(1) <= 1:
            return self._adjacent_delta_feature(frame_features)

        pair_len = (frame_features.size(1) // 2) * 2
        if pair_len < 2:
            return self._adjacent_delta_feature(frame_features)

        pairs = frame_features[:, :pair_len, :].reshape(
            frame_features.size(0), pair_len // 2, 2, frame_features.size(-1))
        pair_deltas = pairs[:, :, 1, :] - pairs[:, :, 0, :]
        mean_delta = pair_deltas.mean(dim=1)
        first_delta = pair_deltas[:, 0, :]
        last_delta = pair_deltas[:, -1, :]
        mean_abs_delta = pair_deltas.abs().mean(dim=1)
        return torch.cat(
            [mean_delta, first_delta, last_delta, mean_abs_delta], dim=-1)

    def _temporal_delta_logits(self, frame_features):
        if self.temporal_delta_cfg.get("detach_visual", False):
            frame_features = frame_features.detach()

        if self.temporal_delta_mode == "edge":
            delta_feature = self._edge_delta_feature(frame_features)
        elif self.temporal_delta_mode == "pair":
            delta_feature = self._pair_delta_feature(frame_features)
        else:
            delta_feature = self._adjacent_delta_feature(frame_features)

        return self.temporal_delta_head(delta_feature.float())

    def forward(self, image, return_attention=False, return_source=False):
        tokenized_prompts = self.tokenized_prompts  # (num_classes, token_len)
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()

        # Encode visual features
        image_features, context_tokens, attn, source = self.image_encoder(image.type(self.dtype),
                                                                          return_layer_num=self.return_layer_num,
                                                                          return_attention=return_attention,
                                                                          return_source=return_source)

        # Now take the mean along the temporal direction with last layer cls tokens
        image_features_mean = image_features[:, -1, ...].mean(dim=1, keepdim=False)
        image_features_mean = image_features_mean / image_features_mean.norm(dim=-1, keepdim=True)  # [b, 512]

        # Instance-conditional prompts
        logits = []
        context_tokens = context_tokens[:, :len(self.prompt_generation_layer_level)]
        for i in range(context_tokens.size(0)): # batch iteration
            text_features = self.text_encoder(prompts=prompts,
                                              tokenized_prompts=tokenized_prompts,
                                              im_features=context_tokens[i, ...])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * image_features_mean[i] @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)    # [b, n_cls]

        if self.use_temporal_delta:
            frame_features = image_features[:, -1, ...]
            delta_logits = self._temporal_delta_logits(frame_features)
            logits = logits + self.temporal_delta_alpha * delta_logits

        return {"logits": logits,
                "attention": attn,
                "source": source}
