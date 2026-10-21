# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from torch import nn
from torch import Tensor
from typing import Callable

from .loss_functional import contrastive_loss, kld_loss, mse_loss
from vlp.core.registry import VLP_LOSSES


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class LossBase(nn.Module):
    """Base class for losses and metrics with config support.
    Do not inherit from this from nn.Module because it can cause issues with Distributed Training.
    """

    def __init__(self, config: dict = None) -> None:
        super().__init__()
        self._config = AttributeDict(config or {})

    def compute_loss(self, **kwargs) -> Tensor:
        """Forward pass that merges config with runtime arguments."""
        # Merge config with kwargs, with kwargs taking precedence
        merged_kwargs = {**self._config, **kwargs}
        return self._loss_fn(**merged_kwargs)

    @property
    def _loss_fn(self) -> Callable:
        """Wrapped base loss function, e.g. l1_loss, mse_loss, etc."""
        raise NotImplementedError("Subclasses must implement _loss_fn property")


@VLP_LOSSES.register_module()
class ContrastiveLoss(LossBase):
    """Contrastive loss with config support."""

    def __init__(self, config: dict = None):
        super().__init__(config)

    @property
    def _loss_fn(self) -> Callable:
        return contrastive_loss


@VLP_LOSSES.register_module()
class MSELoss(LossBase):
    def __init__(self, config: dict = None):
        super().__init__(config)

    @property
    def _loss_fn(self) -> Callable:
        return mse_loss

    def compute_loss(self, input: Tensor, target: Tensor, **kwargs):
        merged_kwargs = {**self._config, **kwargs}
        return self._loss_fn(input=input, target=target, **merged_kwargs)
