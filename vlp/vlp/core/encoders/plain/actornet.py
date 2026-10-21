# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""ActorNet Temporal Convolution Network (TCN) Seq2Vec component. Source: LaneGCN model in https://arxiv.org/abs/2007.13732"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vlp.core.utils.activation import get_activation
from vlp.core.utils.normalization import get_normalization
from vlp.core.encoders.plain.iseq2vec import ISeq2VecModule


from torch import nn


class _Res1d(nn.Module):
    """1D Residual block used in the temporal CNN encoder.

    Args:
        in_features: Number of input features `H`.
        out_features: Number of output features `F`.
        stride: Stride value for the first convolutional layer.
        activation: Activation function string to resolve to nn.Module.
        normalization: Normalization function string to resolve to nn.Module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        activation: str = "elu",
        normalization: Optional[str] = "group_norm_1group",
    ) -> None:
        super().__init__()
        self._conv1 = _Conv1dNormalised(
            in_features=in_features,
            out_features=out_features,
            stride=stride,
            activation=activation,
            normalization=normalization,
        )
        self._conv2 = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self._activation = get_activation(activation=activation)
        if normalization:
            self._normalization1 = get_normalization(normalization=normalization, num_features=out_features)
            self._normalization2 = get_normalization(normalization=normalization, num_features=out_features)
        else:
            self._normalization1 = nn.Identity()
            self._normalization2 = nn.Identity()

        equal_features = in_features == out_features
        if stride != 1 or not equal_features:
            self._downsample: Optional[nn.Module] = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self._downsample = None

    def forward(self, x_NHT: torch.Tensor) -> torch.Tensor:
        """The forward pass.

        Args:
            x_NHT: Input tensor of shape (`N`, `H`, `T`).

        Returns:
            Output tensor of shape (`N`, `F`, `S`).

        Dimensions:
            * `N`: Batch size (e.g. can be number of agents, polylines).
            * `H`: Number of input features.
            * `F`: Number of output features.
            * `T`: Length of the input sequence.
            * `S`: Length of the output sequence.
        """
        identity_NHT = x_NHT

        out_NFS = self._conv1(x_NHT)

        out_NFS = self._conv2(out_NFS)
        out_NFS = self._normalization1(out_NFS)

        if self._downsample is not None:
            identity_NFS = self._downsample(x_NHT)
            identity_NFS = self._normalization2(identity_NFS)
            out_NFS += identity_NFS
        else:
            # F, S == H, T
            out_NFS += identity_NHT

        out_NFS = self._activation(out_NFS)
        return out_NFS


class _Conv1dNormalised(nn.Module):
    """Wrapper class for a 1D convolutional layer followed by normalization and activation.

    Args:
        in_features: Number of input features `H`.
        out_features: Number of output features `F`.
        stride: Stride value for the convolutional layer.
        activation: Activation function string to resolve to nn.Module.
        normalization: Normalization function string to resolve to nn.Module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        activation: str = "elu",
        normalization: Optional[str] = "group_norm_1group",
    ) -> None:
        super().__init__()
        self._conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self._activation = get_activation(activation=activation)
        if normalization:
            self._normalization = get_normalization(normalization=normalization, num_features=out_features)
        else:
            self._normalization = nn.Identity()

    def forward(self, x_NHT: torch.Tensor) -> torch.Tensor:
        """Forward of the 1D convolutional layer.

        Args:
            x_NHT: Input tensor of shape (`N`, `H`, `T`).

        Returns:
            Output tensor of shape (`N`, `F`, `T`).

        Dimensions:
            * `N`: Batch size (e.g. can be number of agents, polylines).
            * `T`: Length of the input and output sequence.
            * `H`: Number of input features.
            * `F`: Number of output features.
        """
        y_NFT = self._conv(x_NHT)
        y_NFT = self._normalization(y_NFT)
        return self._activation(y_NFT)


class ActorNetTCNEncoder(ISeq2VecModule):
    """Temporal (1D) CNN Encoder reproduced from the ActorNet model in LaneGCN: https://arxiv.org/abs/2007.13732.

    Args:
        in_features: Number of input features `H` at each timestep, e.g. x, y, padding.
        out_features: Number of output features `F`.
        strides: Stride values to reduce time dimension in conv layers.
        activation: Activation function string to resolve to nn.Module.
        normalization: Normalization function string to resolve to nn.Module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strides: Tuple[int, int] = (2, 2),
        activation: str = "elu",
        normalization: Optional[str] = "group_norm_1group",
    ) -> None:
        super().__init__(in_features=in_features, out_features=out_features)
        if not in_features >= 1:
            raise ValueError("number of input dimensions must be positive")
        if not out_features >= 1:
            raise ValueError("number of output dimensions must be positive")

        s1, s2 = strides

        feat_small = max(1, out_features // 4)
        feat_medium = max(1, out_features // 2)
        feat_full = out_features

        self._res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    _Res1d(
                        in_features=in_features,
                        out_features=feat_small,
                        activation=activation,
                        normalization=normalization,
                    ),
                    _Res1d(
                        in_features=feat_small,
                        out_features=feat_small,
                        activation=activation,
                        normalization=normalization,
                    ),
                ),
                nn.Sequential(
                    _Res1d(
                        in_features=feat_small,
                        out_features=feat_medium,
                        stride=s1,
                        activation=activation,
                        normalization=normalization,
                    ),
                    _Res1d(
                        in_features=feat_medium,
                        out_features=feat_medium,
                        activation=activation,
                        normalization=normalization,
                    ),
                ),
                nn.Sequential(
                    _Res1d(
                        in_features=feat_medium,
                        out_features=feat_full,
                        stride=s2,
                        activation=activation,
                        normalization=normalization,
                    ),
                    _Res1d(
                        in_features=feat_full,
                        out_features=feat_full,
                        activation=activation,
                        normalization=normalization,
                    ),
                ),
            ]
        )

        self._temp_blocks = nn.ModuleList(
            [
                _Conv1dNormalised(
                    in_features=feat_small,
                    out_features=feat_full,
                    activation=activation,
                    normalization=normalization,
                ),
                _Conv1dNormalised(
                    in_features=feat_medium,
                    out_features=feat_full,
                    activation=activation,
                    normalization=normalization,
                ),
                _Conv1dNormalised(
                    in_features=feat_full,
                    out_features=feat_full,
                    activation=activation,
                    normalization=normalization,
                ),
            ]
        )

        self._output = _Res1d(
            in_features=feat_full,
            out_features=feat_full,
            activation=activation,
            normalization=normalization,
        )

    def forward(
        self,
        x_NTH: Tensor,
    ) -> Tensor:
        """Forward of `ActorNetTCNEncoder`.

        Args:
            x_NTH: Input tensor of shape (`N`, `T`, `H`).

        Returns:
            Sequence embedding of shape (`N`, `F`).

        Dimensions:
            * `N`: Batch size (e.g. can be number of agents, polylines).
            * `T`: Length of the input sequence.
            * `H`: Number of input features.
            * `F`: Number of output features.
        """
        x_NHT = x_NTH.transpose(1, 2)  # switch order of num_timesteps and num_input_features
        x_NCT = x_NHT

        res_block_out = []
        for res_block in self._res_blocks:
            x_NCT = res_block(x_NCT)
            res_block_out.append(x_NCT)

        blocks_iter = iter(zip(self._temp_blocks[::-1], res_block_out[::-1]))

        temp_block, res_out_Nct = next(blocks_iter)
        u_NFt = temp_block(res_out_Nct)

        for temp_block, res_out_Nct in blocks_iter:
            v_NFt = temp_block(res_out_Nct)
            u_NFt = v_NFt + F.interpolate(u_NFt, size=v_NFt.shape[-1], mode="linear")

        out_NF = self._output(u_NFt)[:, :, -1]
        return out_NF


if __name__ == "__main__":
    # Define input parameters
    batch_size = 3  # Number of trajectories
    sequence_length = 6  # Number of timesteps
    input_features = 2  # x, y coordinates
    output_features = 256
    x_NTH = torch.rand(batch_size, sequence_length, input_features)

    # Initialize
    encoder = ActorNetTCNEncoder(
        in_features=input_features,
        out_features=output_features,
        strides=(2, 2),
        activation="elu",
        normalization="group_norm_1group",
    )

    output = encoder(x_NTH)

    # Output shape: (B, F) -> (B, output_features)
    print("Input shape:", x_NTH.shape)
    print("Output shape:", output.shape)
