# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from torch import Tensor

from vlp.core.registry import VLP_MODULES
from vlp.core.encoders.base import DataEncoder


@VLP_MODULES.register_module()
class IdentityBEVEncoder(DataEncoder):
    """Identity BEV features encoder for VLP tasks. Returns the input BEV features."""

    def __init__(self, embed_dims: int = 256):
        super().__init__(embed_dims)

    def get_features(self, bev_embed: Tensor, **kwargs) -> Tensor:
        """Returns the input BEV features as is.

        Args:
            bev_embed: [B, HW, C] BEV features

        Returns:
            Tensor: [B, HW, C] BEV features
        """
        return {"feats": bev_embed}
