# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from vlp.core.registry import VLP_MODULES
from vlp.core.prediction_models.heads import FlexibleMLP
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import LiDARInstanceLines
from vlp.core.encoders.base import DataEncoder
from vlp.vlp.utils import to_tensor
import copy
from torch import nn


@VLP_MODULES.register_module()
class MLPEncoder_SLP(DataEncoder):
    """Simple MLP encoder for ego trajectories with optional extra modalities like ego_cmd.
    It consists of separate encoders for ego-agent and ego-lane features using linear layers
    with optional activation and normalization. Takes only future trajectory.

    Args:
        input_dim (int): Dimension of the input trajectory features (e.g., 2 for x, y coordinates).
        embed_dims (int): Dimension of the output feature embeddings.
        traj_ts (int): Number of time steps in the trajectory.
        egoagent_lin_layers (List[Dict]): List of dictionaries defining linear layers for ego-agent encoding.
                                         Contains keys like 'dim', 'norm', and 'activation'.
        egolane_lin_layers (List[Dict]): List of dictionaries defining linear layers for ego-lane encoding.
                                        Contains keys like 'dim', 'norm', and 'activation'.
        extra_modalities (Dict[str, bool]): Dictionary specifying which extra modalities to use (e.g., ego_cmd).
        extra_modalities_encode_type (str): Encoding type for extra modalities (e.g., "onehot_early_concat").

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - egoagent_feat: Encoded ego-agent features of shape (B, embed_dims)
            - egolane_feat: Encoded ego-lane features of shape (B, embed_dims)

    """

    def __init__(
        self,
        input_dim: int = 2,  # x, y trajectory coords
        embed_dims: int = 256,
        traj_ts: int = 6,
        egoagent_lin_layers: List[Dict] = [{"dim": 112, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}],
        egolane_lin_layers: List[Dict] = [{"dim": 112, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}],
        extra_modalities: Dict[str, bool] = {},
        extra_modalities_encode_type: str = "",
    ) -> None:

        super().__init__(embed_dims=embed_dims)
        self.input_dim = input_dim
        self.traj_ts = traj_ts

        self.use_extra_modalities = True if True in extra_modalities.values() else False
        self.extra_modalities = extra_modalities

        if self.use_extra_modalities:
            assert extra_modalities_encode_type in [
                "onehot_early_concat"
            ], f"Wrong encoding type for extra modalities: {extra_modalities_encode_type}"
            # Define command and agent label mappings
            self.cmd_map = {0: "turning right", 1: "turning left", 2: "going straight"}
            extra_input_dim = self._get_extra_input_dim()
        else:
            extra_input_dim = 0

        self.full_input_dim = input_dim + extra_input_dim
        self.egoagent_encoder = FlexibleMLP(
            input_dim=(self.full_input_dim * traj_ts),
            lin_layers=egoagent_lin_layers,
            output_dim=embed_dims,
        )

        self.egolane_encoder = FlexibleMLP(
            input_dim=(self.full_input_dim * traj_ts),
            lin_layers=egolane_lin_layers,
            output_dim=embed_dims,
        )

    def _get_extra_input_dim(self) -> int:
        """Returns the number of input features for the encoder from extra modalities."""
        in_features = 0

        if self.use_extra_modalities:
            for modality, enabled in self.extra_modalities.items():
                if enabled:
                    if modality == "ego_cmd":
                        in_features += len(self.cmd_map)
                    else:
                        raise ValueError(f"Unknown extra modality: {modality}")
        return in_features

    def _get_ego_cmd_one_hot(
        self,
        ego_fut_cmd: torch.Tensor,
    ) -> Union[torch.Tensor, None]:
        """Convert ego command into one-hot encoding.

        Args:
            ego_fut_cmd: Tensor containing future commands for the ego vehicle.

        Returns:
            ego_one_hot: One-hot encoded tensor for the ego vehicle commands. [B, one_hot_dim] or None

        """
        device = ego_fut_cmd.device
        ego_one_hot = []

        assert self.extra_modalities.keys() <= {"ego_cmd"}

        if self.extra_modalities.get("ego_cmd", False):
            # ego command
            ego_cmd_one_hot = ego_fut_cmd[:, 0, 0]  # [B, 3]
            # append
            ego_one_hot.append(ego_cmd_one_hot)

        # Concatenate features if they exist
        ego_one_hot = torch.cat(ego_one_hot, dim=1).to(device) if len(ego_one_hot) > 0 else None

        return ego_one_hot

    def get_features(
        self,
        ego_fut_trajs: torch.Tensor,
        gt_labels_3d: List[torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features for ego trajectories using MLP encoding with optional extra modalities.
        [B, 1, ts, input_dim] -> [B, ts * (input_dim + extra_dim)] -> MLP -> [B, embed_dims]

        Args:
            ego_fut_trajs: Ego trajectories of shape (B, 1, T, 2).
            gt_labels_3d: List of agent labels for each batch.
            ego_fut_cmd: Ego future commands for extra modality encoding.
            gt_attr_labels: List of agent future trajectories for each batch.
            map_gt_bboxes_3d: Lane bounding boxes for each batch.

        Returns:
            Tuple containing:
            - egoagent_feat: Encoded ego-agent features of shape (B, embed_dims)
            - egolane_feat: Encoded ego-lane features of shape (B, embed_dims)
        """
        assert self.use_extra_modalities, "Extra modalities are not enabled in the configuration."

        # Convert ego command to one-hot encoding
        ego_one_hot = self._get_ego_cmd_one_hot(ego_fut_cmd)

        device = ego_fut_trajs.device
        self.egoagent_encoder.to(device)
        self.egolane_encoder.to(device)

        B = ego_fut_trajs.shape[0]

        # Encode ego trajectories
        ego_fut_trajs = ego_fut_trajs[:, 0, :, :]  # Shape: (B, T, 2)
        traj_input = self._merge_traj_with_one_hot(ego_fut_trajs, ego_one_hot)  # (B, T, 2 + one_hot_dim)

        # Reshape and encode ego input
        traj_input = traj_input.reshape(B, -1)  # Shape: (B, T * (2 + one_hot_dim))
        assert (
            traj_input.shape[1] == self.full_input_dim * self.traj_ts
        ), f"Expected input shape (B, T * (2 + one_hot_dim)), got {ego_fut_trajs.shape}."

        egoagent_feat = self.egoagent_encoder(traj_input)  # Shape: (B, embed_dims)
        egolane_feat = self.egolane_encoder(traj_input)  # Shape: (B, embed_dims)

        return egoagent_feat, egolane_feat

    def _merge_traj_with_one_hot(
        self,
        traj: torch.Tensor,
        one_hot: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        """
        Merges trajectory data with one-hot encoded features.

        Args:
            traj: Trajectory tensor of shape (B, T, 2).
            one_hot: One-hot encoded tensor of shape (B, one_hot_dim).

        Returns:
            If one_hot is not None: Merged tensor of shape (B, T, 2 + one_hot_dim). Otherwise just traj tensor.
        """
        if one_hot is None:
            return traj

        B, T, _ = traj.shape
        assert one_hot.shape[0] == B, "One-hot encoding shape mismatch with batch size."
        one_hot = one_hot.unsqueeze(1).expand(-1, T, -1)  # Shape: (B, T, one_hot_dim)
        assert one_hot.shape[:2] == traj.shape[:2], "One-hot encoding shape mismatch with trajectory shape."
        merged_traj = torch.cat([traj, one_hot], dim=-1)  # Shape: (B, T, 2 + one_hot_dim)

        return merged_traj
