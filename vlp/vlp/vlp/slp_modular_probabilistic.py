# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional, Union
from vlp.vlp.utils import gather_tensor
from vlp.core.registry import VLP_MODULES, VLP_LOSSES
from mmcv.utils import get_logger
from mmcv import ConfigDict
from torch.distributions import Distribution

logger = get_logger("SLP_Prob")


@VLP_MODULES.register_module()
class SLP_Planner_Probabilistic(nn.Module):
    """Generic Vision-Language Planning framework. SLP Part.
    Performs alignement between ego <-> agent and ego <-> lanes interaction queries with their corresponding target features.
    """

    def __init__(self, target_encoder: ConfigDict, loss: ConfigDict, **kwargs) -> None:
        """Initialize SLP Planner with flexible encoder architecture.

        Args:
            target_encoder: Encoder for target features (e.g., LLMEncoder, ImageEncoder, etc.)
            query_encoder: Wrapper to process ego-queries
            feature_aligner: Feature alignment module for computing losses
            pc_range: Point cloud range for bbox normalization
            embed_dims: Dimension of embedding space
            contra_logit_scale_init: Initial value for logit scaling parameter. Used for contrastive losses only.
            **kwargs: Additional arguments passed to encoders
        """
        super().__init__()

        # Initialize encoders
        self.target_encoder = VLP_MODULES.build(target_encoder)
        self.loss = VLP_LOSSES.build(loss)

        # Log trainable parameters
        logger.info(f"[SLP] Loaded Target Encoder: {self.target_encoder.__class__.__name__}")
        logger.info(f"[SLP] Target Encoder Trainable Params: \n  {self.get_trainable_params_list(self.target_encoder)}")
        logger.info(f"[SLP] Loaded Loss: {self.loss.__class__.__name__}")

    def get_trainable_params_list(self, module: nn.Module) -> List[str]:
        """Get a list of trainable parameters in the module."""
        return [name for name, param in module.named_parameters() if param.requires_grad]

    def get_target_dist(self, **kwargs) -> Union[torch.Tensor, Distribution]:
        """Get the target distribution from the target encoder."""
        return self.target_encoder.get_distribution(**kwargs)

    def compute_query_loss(
        self, query_target_dist: Distribution, query_source_dist: Distribution, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute BEV alignment losses.

        Returns:
            Dictionary of BEV losses
        """
        loss = self.loss(dist1=query_target_dist, dist2=query_source_dist, **kwargs)
        return loss

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device
