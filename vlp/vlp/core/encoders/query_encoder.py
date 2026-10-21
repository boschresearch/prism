# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import copy
import numpy as np
import cv2
import torch
from typing import List, Optional, Tuple

from vlp.core.registry import VLP_MODULES
from vlp.core.encoders.base import DataEncoder
from vlp.vlp.utils import to_tensor


@VLP_MODULES.register_module()
class QueryEncoder(DataEncoder):
    """Dummy encoder to process VLP queries for SLP tasks."""

    def __init__(self, embed_dims: int = 256):
        super().__init__(embed_dims)

    def get_features(
        self, egoagent_query: torch.Tensor, egolane_query: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features for ego agent and lanes from the input queries.

        Args:
            egoagent_query: [B, 1, D] Ego <-> agents interaction query features.
            egolane_query:  [B, 1, D] Ego <-> map interaction query features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Ego agent features [B, D]
                - Ego lane features  [B, D]
        """
        egoagent_query = egoagent_query.squeeze(1)
        egolane_query = egolane_query.squeeze(1)

        assert (
            egoagent_query.shape == egolane_query.shape
        ), f"Ego agent and lane queries must have the same shape. Got {egoagent_query.shape} and {egolane_query.shape}."

        return egoagent_query, egolane_query
