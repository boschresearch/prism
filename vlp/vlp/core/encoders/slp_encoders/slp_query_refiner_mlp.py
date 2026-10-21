# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from vlp.core.registry import VLP_MODULES
from vlp.core.prediction_models.heads import FlexibleMLP
from vlp.core.encoders.base import DataEncoder
from vlp.vlp.utils import to_tensor
import copy
from torch import nn


@VLP_MODULES.register_module()
class SLPQueryRefinerMLP(DataEncoder):
    """SLP query refiner that uses MLP to predict delta corrections for ego queries.

    Takes ego trajectory data (with optional command modalities) and predicts delta
    corrections to refine existing ego-agent and ego-lane query features. The MLP
    outputs deltas for both query types which are added to the original queries.
    """

    def __init__(
        self,
        input_dim: int = 2,  # x, y trajectory coords
        embed_dims: int = 256,
        traj_ts: int = 6,
        gt_data_encoder_lin_layers: List[Dict] = [
            {"dim": 256, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}
        ],
        refiner_lin_layers: List[Dict] = [{"dim": 256, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}],
        extra_modalities: Dict[str, bool] = {},
    ) -> None:
        """Initialize query refiner with trajectory encoder and dual refinement MLPs.

        Architecture: trajectory → encoder → split features → concat with queries → refine

        Args:
            input_dim: Trajectory coordinate dimension (default: 2 for x,y)
            embed_dims: Output dimension for each refined query type
            traj_ts: Trajectory sequence length
            gt_data_encoder_lin_layers: Trajectory encoder MLP config (outputs embed_dims * 2)
            refiner_lin_layers: Query refiner MLP config (input: query + encoded features)
            extra_modalities: Additional inputs {'ego_cmd': bool} for command encoding
        """

        super().__init__(embed_dims=embed_dims)
        self.input_dim = input_dim
        self.traj_ts = traj_ts

        self.use_extra_modalities = True if True in extra_modalities.values() else False
        self.extra_modalities = extra_modalities

        if self.use_extra_modalities:
            # Define command and agent label mappings
            self.cmd_map = {0: "turning right", 1: "turning left", 2: "going straight"}
            extra_input_dim = self._get_extra_input_dim()
        else:
            extra_input_dim = 0

        self.full_input_dim = input_dim + extra_input_dim

        self.gt_data_encoder = FlexibleMLP(
            input_dim=(self.full_input_dim * traj_ts),
            lin_layers=gt_data_encoder_lin_layers,
            output_dim=(embed_dims * 2),
        )
        self.egoagent_query_refiner = FlexibleMLP(
            input_dim=(embed_dims * 2),
            lin_layers=refiner_lin_layers,
            output_dim=embed_dims,
        )
        self.egolane_query_refiner = FlexibleMLP(
            input_dim=(embed_dims * 2),
            lin_layers=refiner_lin_layers,
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
        egoagent_query: torch.Tensor,
        egolane_query: torch.Tensor,
        ego_fut_trajs: torch.Tensor,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine queries by encoding trajectory features and concatenating with original queries.

        Process: encode trajectory → split features → concat with queries → refine via MLPs

        Args:
            egoagent_query: Original ego-agent query [B, 1, embed_dims]
            egolane_query: Original ego-lane query [B, 1, embed_dims]
            ego_fut_trajs: Ego trajectory [B, 1, T, 2]
            ego_fut_cmd: Ego commands [B, 1, 1, 3] (one-hot)
            **kwargs: Additional unused arguments

        Returns:
            Tuple of refined queries:
                - egoagent_query_refined [B, embed_dims]
                - egolane_query_refined [B, embed_dims]
        """

        assert self.use_extra_modalities, "Extra modalities are not enabled in the configuration."

        device = ego_fut_trajs.device
        B = ego_fut_trajs.shape[0]

        self.gt_data_encoder.to(device)
        self.egoagent_query_refiner.to(device)
        self.egolane_query_refiner.to(device)

        # Prepare GT input: trajectories and one-hot ego command (optional)
        ego_fut_trajs = ego_fut_trajs[:, 0, :, :]  # Shape: (B, T, 2)
        ego_one_hot = self._get_ego_cmd_one_hot(ego_fut_cmd)
        traj_input = self._merge_traj_with_one_hot(ego_fut_trajs, ego_one_hot)  # (B, T, 2 + one_hot_dim)
        traj_input = traj_input.reshape(B, -1)  # Shape: (B, T * (2 + one_hot_dim))
        assert (
            traj_input.shape[1] == self.full_input_dim * self.traj_ts
        ), f"Expected input shape (B, T * (2 + one_hot_dim)), got {ego_fut_trajs.shape}."

        # Encode GT input into features for two queries
        gt_data_features = self.gt_data_encoder(traj_input)  # Shape: (B, embed_dims * 2)
        assert gt_data_features.shape == (B, egoagent_query.shape[-1] + egolane_query.shape[-1])

        # Compbine original queries with GT features
        egoagent_gt_featrues, egolane_gt_features = torch.split(
            gt_data_features, [egoagent_query.shape[-1], egolane_query.shape[-1]], dim=-1
        )
        egoagent_query_refined = torch.cat(
            [egoagent_query.squeeze(1), egoagent_gt_featrues], dim=-1
        )  # Shape: (B, embed_dims * 2)
        egolane_query_refined = torch.cat(
            [egolane_query.squeeze(1), egolane_gt_features], dim=-1
        )  # Shape: (B, embed_dims * 2)

        # Refine combined input using MLP
        egoagent_query_refined = self.egoagent_query_refiner(egoagent_query_refined)
        egolane_query_refined = self.egolane_query_refiner(egolane_query_refined)

        return egoagent_query_refined, egolane_query_refined

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


def test():
    """Run a simple test to verify the SLPQueryRefinerMLP functionality."""
    input_dim = 2
    embed_dims = 256
    traj_ts = 6
    gt_data_encoder_lin_layers = [{"dim": 256, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}]
    refiner_lin_layers = [{"dim": 256, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}]
    extra_modalities = {"ego_cmd": True}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    refiner = SLPQueryRefinerMLP(
        input_dim=input_dim,
        embed_dims=embed_dims,
        traj_ts=traj_ts,
        gt_data_encoder_lin_layers=gt_data_encoder_lin_layers,
        refiner_lin_layers=refiner_lin_layers,
        extra_modalities=extra_modalities,
    ).to(device)

    # Dummy data
    B = 4
    egoagent_query = torch.randn(B, 1, embed_dims).to(device)
    egolane_query = torch.randn(B, 1, embed_dims).to(device)
    ego_fut_trajs = torch.randn(B, 1, traj_ts, input_dim).to(device)
    ego_fut_cmd = torch.zeros(B, 1, 1, 3).to(device)

    refined_agent_query, refined_lane_query = refiner.get_features(
        egoagent_query=egoagent_query, egolane_query=egolane_query, ego_fut_trajs=ego_fut_trajs, ego_fut_cmd=ego_fut_cmd
    )

    print("Refined Ego-Agent Query Shape:", refined_agent_query.shape)
    print("Refined Ego-Lane Query Shape:", refined_lane_query.shape)
    print(f"model: {refiner}")


if __name__ == "__main__":
    test()
