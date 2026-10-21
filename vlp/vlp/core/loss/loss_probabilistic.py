# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from torch import nn, Tensor
from typing import Callable, Dict, Optional, Any
from vlp.core.registry import VLP_LOSSES
from torch.distributions import Distribution


class ProbLossBase(nn.Module):
    """
    Generic wrapper for probabilistic losses with MMCV config support.

    Args:
        loss_fn (Callable): the actual loss function, e.g. kld_loss(dist1, dist2, **kwargs)
        name (str): base name for the loss key in output dict
        weight (float): scaling factor for the loss
        config (dict, optional): extra configuration for the loss
    """

    def __init__(
        self,
        loss_fn: Callable,
        name: str,
        weight: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.name = name
        self.weight = weight
        self.config = config or {}

    def forward(
        self,
        dist1: Distribution,
        dist2: Distribution,
        suffix: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Args:
            dist1, dist2: input distributions
            suffix (str, optional): suffix to append to loss name (e.g. "_aux")
            kwargs: runtime args passed to the loss function

        Returns:
            dict: { "<name>(+suffix)" : loss_value }
        """
        # merge config and runtime kwargs
        merged_kwargs = {**self.config, **kwargs}

        loss_value = self.loss_fn(dist1, dist2, **merged_kwargs)
        weighted_loss = loss_value * self.weight

        key = self.name + ("_" + suffix if suffix else "")
        return {key: weighted_loss}


# ===== Example Loss Implementations =====


def kld_loss(dist1, dist2, reduction="mean"):
    """KL divergence between two distributions."""
    kl = torch.distributions.kl_divergence(dist1, dist2)
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


@VLP_LOSSES.register_module()
class KLDLoss(ProbLossBase):
    def __init__(self, name="kld_loss", weight=1.0, config=None):
        super().__init__(loss_fn=kld_loss, name=name, weight=weight, config=config)
