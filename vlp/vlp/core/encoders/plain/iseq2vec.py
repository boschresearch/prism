# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Interface for Seq2Vec components."""

import abc
from abc import ABCMeta
from typing import Optional

from torch import Tensor, nn


def assert_feature_size_positive(dim: int, name: str = "Not Provided") -> None:
    """Check if the feature dimension is a positive integer.

    Args:
        dim: The feature dimensions to check.
        name: The name of the features to include in the error message.
    """
    if not isinstance(dim, int):
        raise TypeError(
            f"The type of feature dimensions for features: `{name}` must be an integer, got type: {type(dim)}."
        )
    if dim <= 0:
        raise ValueError(f"The dimensions of features: `{name}` must be a positive integer, got value: {dim}")


class ISeq2VecModule(nn.Module, metaclass=ABCMeta):
    """Interface of sequence-to-vector modules.

    An `ISeq2VecModule` is designed to process sequences of feature vectors of shape (`N`, `T`, `H`) and produce
    an embedded sequence as a single feature vector of shape (`N`, `F`).

    Args:
        in_features: Number of input features `H`.
        out_features: Number of output features `F`.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        assert_feature_size_positive(in_features, name="in_features")
        assert_feature_size_positive(out_features, name="out_features")

        super().__init__()
        self._in_features = in_features
        self._out_features = out_features

    @property
    def in_features(self) -> int:
        """Number of input features `H`."""
        return self._in_features

    @property
    def out_features(self) -> int:
        """Number of output features `F`."""
        return self._out_features

    @abc.abstractmethod
    def forward(
        self,
        x_NTH: Tensor,
        x_mask_NT: Optional[Tensor],
    ) -> Tensor:
        """Forward of `Seq2Vec` modules.

        Args:
            x_NTH: Input tensor of shape (`N`, `T`, `H`).
            x_mask_NT: Mask of shape (`N`, `T`). Defaults all elements to `True` if `None`.

        Returns:
            Sequence embedding of shape (`N`, `F`).

        Dimensions:
            * `N`: Batch size (e.g. can be number of agents, polylines).
            * `T`: Length of the input sequence.
            * `H`: Number of input feature.
            * `F`: Number of output features.
        """
        pass
