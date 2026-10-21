# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from abc import ABC, abstractmethod
from typing import Dict, Union
from torch import nn
from torch.distributions import Distribution


class DataEncoder(ABC, nn.Module):
    """Base class for all data encoders to be used in VLP tasks."""

    def __init__(self, embed_dims: int = 256):
        super().__init__()
        self.embed_dims = embed_dims

    @abstractmethod
    def get_features(self, **kwargs) -> Dict:
        """Main method to extract features from input data

        Returns:
            torch.Tensor: Encoded features
        """
        pass


class ProbabilisticDataEncoder(ABC, nn.Module):
    """Base class for all data encoders to be used in VLP tasks."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_distribution(self, **kwargs) -> Union[Distribution, Dict]:
        """Main method to extract distribution parameters from input data"""
        pass
