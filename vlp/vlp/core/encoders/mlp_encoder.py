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
class MLPEncoder(DataEncoder):
    """Simple MLP encoder trajectories and, optionally, for other data modalities, e.g. ego_cmd, labels.
        It consist of a series of linear layers with optional activation and normalization.

    Args:
        input_dim (int): Dimension of the input features (e.g., 2 for x, y coordinates).
        embed_dims (int): Dimension of the output feature embeddings.
        traj_ts (int): Number of time steps in the trajectory.
        bev_lane_points (int): Number of points to sample along each lane.
        traj_lin_layers (List[Dict]): List of dictionaries defining linear layers for trajectory encoding.
                                      Contains keys like 'dim', 'norm', and 'activation'.
        lane_lin_layers (List[Dict]): List of dictionaries defining linear layers for lane encoding.
                                    Contains keys like 'dim', 'norm', and 'activation'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Combined features: [(B + BM++ + BL++), embed_dims],
                                 (1 - ego, M - agents, L - lanes, ++ - varying size)
            - Agent batch indices
            - Lane batch indices

    """

    def __init__(
        self,
        input_dim: int = 2,  # x, y trajectory coords
        embed_dims: int = 256,
        traj_ts: int = 6,
        bev_lane_points: int = 200,
        traj_lin_layers: List[Dict] = [{"dim": 112, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}],
        lane_lin_layers: List[Dict] = [{"dim": 112, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}],
        extra_modalities: Dict[str, bool] = {},
        extra_modalities_encode_type: str = "",
    ) -> None:

        super().__init__(embed_dims=embed_dims)
        self.input_dim = input_dim
        self.bev_lane_points = bev_lane_points
        self.traj_ts = traj_ts

        self.use_extra_modalities = True if True in extra_modalities.values() else False
        self.extra_modalities = extra_modalities

        if self.use_extra_modalities:
            assert extra_modalities_encode_type in [
                "onehot_early_concat"
            ], f"Wrong encoding type for extra modalities: {extra_modalities_encode_type}"
            # Define command and agent label mappings
            self.cmd_map = {0: "turning right", 1: "turning left", 2: "going straight"}
            self.agent_label_map = {
                0: "a car",
                1: "a truck",
                2: "a construction vehicle",
                3: "a bus",
                4: "a trailer",
                5: "a barrier",
                6: "a motorcycle",
                7: "a bicycle",
                8: "a pedestrian",
                9: "a traffic_cone",
            }

            extra_input_dim = self._get_extra_input_dim()
        else:
            extra_input_dim = 0

        self.full_input_dim = input_dim + extra_input_dim
        self.traj_encoder = FlexibleMLP(
            input_dim=(self.full_input_dim * traj_ts),
            lin_layers=traj_lin_layers,
            output_dim=embed_dims,
        )

        self.lane_encoder = FlexibleMLP(
            input_dim=(self.full_input_dim * bev_lane_points),
            lin_layers=lane_lin_layers,
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
                    elif modality == "label":
                        in_features += len(self.agent_label_map) + 2  # +1 for lane label, +1 for ego vehicle label
                    else:
                        raise ValueError(f"Unknown extra modality: {modality}")
        return in_features

    def _extra_gt_to_one_hot(
        self,
        ego_fut_cmd: torch.Tensor,
        gt_labels_3d: List[torch.Tensor],
        map_gt_bboxes_3d: List[LiDARInstanceLines],
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Convert extra gt modalities to one-hot encoding.
        Structure: [ ---Ego-CMD---, ---AGENT-LABEL---];
                    AGENT-LABEL: Original nuScenes agent labels + 1 (lane label) + 1 (ego vehicle label)

        Args:
            ego_fut_cmd: Tensor containing future commands for the ego vehicle.
            gt_labels_3d: List of tensors containing 3D labels for agents in each batch. Length B.
            map_gt_bboxes_3d: List of lane bounding boxes for each batch. Length B.

        Returns:
            Three one-hot encoded tensors:
                - ego_one_hot: One-hot encoded tensor for the ego vehicle commands. [B, one_hot_dim] or None
                - agent_one_hot: One-hot encoded tensor for agent labels. [BM, one_hot_dim] or None
                - lane_one_hot: One-hot encoded tensor for lane labels. [BL, one_hot_dim] or None
        """
        device = ego_fut_cmd.device
        ego_one_hot = []
        agent_one_hot = []
        lane_one_hot = []

        assert self.extra_modalities.keys() <= {
            "ego_cmd",
            "label",
        }, "Extra modalities must be a subset of {'ego_cmd', 'label'}."

        if self.extra_modalities.get("ego_cmd", False):
            # ego command
            ego_cmd_one_hot = ego_fut_cmd[:, 0, 0]  # [B, 3]

            # agent commands (no data available, all zeros)
            num_agents = sum(labels_per_batch.shape[0] for labels_per_batch in gt_labels_3d)
            agent_cmd_one_hot = torch.zeros(
                (num_agents, len(self.cmd_map)), device=ego_fut_cmd.device
            )  # [BM, num_cmds]

            # lane commands (no data available, all zeros)
            num_lanes = sum(len(lane_bboxes_per_batch.instance_list) for lane_bboxes_per_batch in map_gt_bboxes_3d)
            lane_cmd_one_hot = torch.zeros((num_lanes, len(self.cmd_map)), device=ego_fut_cmd.device)
            # [BL, num_cmds]

            # append
            ego_one_hot.append(ego_cmd_one_hot)
            agent_one_hot.append(agent_cmd_one_hot)
            lane_one_hot.append(lane_cmd_one_hot)

        if self.extra_modalities.get("label", False):
            # ego labels (car label - index -1)
            ego_label_one_hot = torch.zeros(
                (ego_fut_cmd.shape[0], len(self.agent_label_map) + 2), device=ego_fut_cmd.device
            )  # [B, num_agent_labels + 2]
            ego_label_one_hot[:, -1] = 1  # Set car label (index 0) for ego

            # agent labels one-hot encoding
            agent_idx = torch.cat(gt_labels_3d, dim=0)  # [BM,]
            agent_label_one_hot = nn.functional.one_hot(
                agent_idx, num_classes=len(self.agent_label_map) + 2  # +2 for lane and ego labels
            ).float()  # [BM, num_agent_labels]

            # lane labels (assign last label to all lanes, index -2)
            num_lanes = sum(len(lane_bboxes_per_batch.instance_list) for lane_bboxes_per_batch in map_gt_bboxes_3d)
            lane_label_one_hot = torch.zeros(
                (num_lanes, len(self.agent_label_map) + 2), device=ego_fut_cmd.device
            )  # [BL, num_agent_labels + 2]
            if num_lanes > 0:
                lane_label_one_hot[:, -2] = 1  # Set -2 dimension to 1 for lanes

            # append
            ego_one_hot.append(ego_label_one_hot)
            agent_one_hot.append(agent_label_one_hot)
            lane_one_hot.append(lane_label_one_hot)

        # Concatenate features if they exist
        ego_one_hot = torch.cat(ego_one_hot, dim=1).to(device) if len(ego_one_hot) > 0 else None
        agent_one_hot = torch.cat(agent_one_hot, dim=1).to(device) if len(agent_one_hot) > 0 else None
        lane_one_hot = torch.cat(lane_one_hot, dim=1).to(device) if len(lane_one_hot) > 0 else None

        return ego_one_hot, agent_one_hot, lane_one_hot

    def get_features(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_extra_modalities:
            return self.get_features_with_extra_modalities(**kwargs)
        else:
            return self.get_features_from_traj(**kwargs)

    def get_features_with_extra_modalities(
        self,
        ego_fut_trajs: torch.Tensor,
        gt_labels_3d: List[torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features for ego, agents, and lanes.
        [B, 1, ts, input_dim] -> [B, ts * input_dim] -> MLP -> [B, embed_dims]


        Args:
            ego_fut_trajs: Ego trajectories of shape (B, 1, T, 2).
            gt_labels_3d: List of agent labels for each batch.
            gt_attr_labels: List of agent future trajectories for each batch.
            map_gt_bboxes_3d: Lane bounding boxes for each batch.
            pc_range: Point cloud range for normalization.

        Returns:
            Tuple containing:
                - Combined features [(B + BM++ + BL++), embed_dims],
                                    (1 - ego, M - agents, L - lanes, ++ - varying size)
                - Agent batch indices
                - Lane batch indices
        """
        assert self.use_extra_modalities, "Extra modalities are not enabled in the configuration."

        # Convert extra modalities to one-hot encoding
        ego_one_hot, agent_one_hot, lane_one_hot = self._extra_gt_to_one_hot(
            ego_fut_cmd, gt_labels_3d, map_gt_bboxes_3d
        )

        device = ego_fut_trajs.device
        self.traj_encoder.to(device)
        self.lane_encoder.to(device)

        B = ego_fut_trajs.shape[0]

        # Encode ego trajectories
        ego_fut_trajs = ego_fut_trajs[:, 0, :, :]  # Shape: (B, T, 2)
        traj_input = self._merge_traj_with_one_hot(ego_fut_trajs, ego_one_hot)  # (B, T, 2 + one_hot_dim)

        # Reshape and encode ego input
        traj_input = traj_input.reshape(B, -1)  # Shape: (B, T * (2 + one_hot_dim))
        assert (
            traj_input.shape[1] == self.full_input_dim * self.traj_ts
        ), f"Expected input shape (B, T * (2 + one_hot_dim)), got {ego_fut_trajs.shape}."
        ego_features = self.traj_encoder(traj_input)  # Shape: (B, embed_dims)

        # Generate agent trajectories
        agent_batch_idx = []
        agent_trajs = []
        for bi in range(B):
            agent_batch_idx.extend([bi] * gt_labels_3d[bi].shape[0])  # Needed for further aggregation from gpus

            agent_fut_traj = copy.deepcopy(gt_attr_labels[bi][:, : self.traj_ts * 2])  # Shape: (M, traj_ts * 2)
            agent_fut_traj = agent_fut_traj.reshape(-1, self.traj_ts, 2)  # Shape: (M, traj_ts, 2)
            agent_trajs.append(agent_fut_traj)

        agent_trajs = (
            torch.cat(agent_trajs, dim=0) if agent_trajs else torch.empty(0, self.traj_ts, 2, device=device)
        )  # [BM, traj_ts, 2]

        # Encode agent trajectories
        if agent_trajs.shape[0] > 0:
            agent_input = self._merge_traj_with_one_hot(agent_trajs, agent_one_hot)  # (BM, fut_ts, 2 + one_hot_dim)
            # Reshape (BM, traj_ts, 2 + one_hot_dim) -> (BM, traj_ts * (2 + one_hot_dim))
            agent_input = agent_input.reshape(agent_input.shape[0], -1)
            assert (
                agent_input.shape[1] == self.full_input_dim * self.traj_ts
            ), f"Expected input shape (BM, traj_ts * (2 + one_hot_dim)), got {agent_input.shape}."
            # Encode agent input
            agent_features = self.traj_encoder(agent_input)  # Shape: (BM, embed_dims)
        else:
            agent_features = torch.empty(0, self.embed_dims, device=device)

        # Encode lane trajectories
        lane_points = []
        lane_batch_idx = []
        if map_gt_bboxes_3d:
            # sample points on every lane from all batches
            # NOTE: might be not the best approach when batch size and num of points sampled from lanes  are large, because some traj encoders will reshape input to [(BL++ * bev_lane_points), 2], leading to first dim being very large.
            for bi, map_bboxes in enumerate(map_gt_bboxes_3d):
                lane_points_per_batch = self._extract_lane_points(map_bboxes)  # (L, bev_lane_points, 2)
                if lane_points_per_batch.shape[0] > 0:
                    lane_points.append(lane_points_per_batch)
                    lane_batch_idx.extend([bi] * lane_points_per_batch.shape[0])

            # if batch has lanes -> encode
            if lane_points:
                lane_points = torch.cat(lane_points, dim=0)  # (BL++, bev_lane_points, 2)
                lane_input = self._merge_traj_with_one_hot(
                    lane_points, lane_one_hot
                )  # (BL++, bev_lane_points, 2 + one_hot_dim)

                # Reshape (BL++, bev_lane_points, 2 + one_hot_dim) -> (BL++, bev_lane_points * (2 + one_hot_dim))
                B, _, _ = lane_input.shape
                lane_input = lane_input.reshape(B, -1)

                assert (
                    lane_input.shape[1] == self.full_input_dim * self.bev_lane_points
                ), f"Expected input shape (BL, bev_lane_points * (2 + one_hot_dim)), got {lane_input.shape}."
                lane_features = self.lane_encoder(lane_input)  # [B, embed_dims]
            else:
                lane_features = torch.empty(0, self.embed_dims, device=device)

        # Combine all features
        combined_features = torch.cat([ego_features, agent_features, lane_features], dim=0)
        return {
            "feats": combined_features,
            "agent_batch_idx": torch.tensor(agent_batch_idx),
            "lane_batch_idx": torch.tensor(lane_batch_idx),
        }

    def get_features_from_traj(
        self,
        ego_fut_trajs: torch.Tensor,
        gt_labels_3d: List[torch.Tensor],
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features for ego, agents, and lanes.
        [B, 1, ts, input_dim] -> [B, ts * input_dim] -> MLP -> [B, embed_dims]


        Args:
            ego_fut_trajs: Ego trajectories of shape (B, 1, T, 2).
            gt_labels_3d: List of agent labels for each batch.
            gt_attr_labels: List of agent future trajectories for each batch.
            map_gt_bboxes_3d: Lane bounding boxes for each batch.
            pc_range: Point cloud range for normalization.

        Returns:
            Tuple containing:
                - Combined features [(B + BM++ + BL++), embed_dims],
                                    (1 - ego, M - agents, L - lanes, ++ - varying size)
                - Agent batch indices
                - Lane batch indices
        """
        device = ego_fut_trajs.device
        self.traj_encoder.to(device)
        self.lane_encoder.to(device)

        B = ego_fut_trajs.shape[0]

        # Encode ego trajectories
        ego_fut_trajs = ego_fut_trajs[:, 0, :, :]  # Shape: (B, T, 2)

        # Reshape
        ego_fut_trajs = ego_fut_trajs.reshape(B, -1)  # Shape: (B, T * 2)
        ego_features = self.traj_encoder(ego_fut_trajs)  # Shape: (B, embed_dims)

        # Generate agent trajectories
        agent_batch_idx = []
        agent_trajs = []
        for bi in range(B):
            agent_batch_idx.extend([bi] * gt_labels_3d[bi].shape[0])  # Needed for further aggregation from gpus

            agent_fut_traj = copy.deepcopy(gt_attr_labels[bi][:, : self.traj_ts * 2])  # Shape: (M, traj_ts * 2)
            agent_fut_traj = agent_fut_traj.reshape(-1, self.traj_ts, 2)  # Shape: (M, traj_ts, 2)
            agent_trajs.append(agent_fut_traj)

        agent_trajs = (
            torch.cat(agent_trajs, dim=0) if agent_trajs else torch.empty(0, self.traj_ts, 2, device=device)
        )  # [BM, traj_ts, 2]

        # Encode agent trajectories
        if agent_trajs.shape[0] > 0:
            # Reshape (BM, traj_ts, 2) -> (BM, traj_ts * 2)
            agent_trajs = agent_trajs.reshape(agent_trajs.shape[0], -1)
            agent_features = self.traj_encoder(agent_trajs)  # Shape: (BM, embed_dims)
        else:
            agent_features = torch.empty(0, self.embed_dims, device=device)

        # Encode lane trajectories
        lane_features = []
        lane_batch_idx = []
        if map_gt_bboxes_3d:
            for bi, map_bboxes in enumerate(map_gt_bboxes_3d):
                lane_points = self._extract_lane_points(map_bboxes)

                # Reshape (B, bev_lane_points, input_dim) -> (B, bev_lane_points * input_dim)
                B, _, _ = lane_points.shape
                lane_points = lane_points.reshape(B, -1)

                lane_feat = self.lane_encoder(lane_points)
                if lane_feat.shape[0] > 0:
                    lane_features.append(lane_feat)
                    lane_batch_idx.extend([bi] * lane_feat.shape[0])
            if lane_features:
                lane_features = torch.cat(lane_features, dim=0)  # Shape: (BL, embed_dims)
            else:
                lane_features = torch.empty(0, self.embed_dims, device=device)

        # Combine all features
        combined_features = torch.cat([ego_features, agent_features, lane_features], dim=0)
        return {
            "feats": combined_features,
            "agent_batch_idx": torch.tensor(agent_batch_idx),
            "lane_batch_idx": torch.tensor(lane_batch_idx),
        }

    def _extract_lane_points(self, map_gt_bboxes_3d) -> torch.Tensor:
        """
        This method samples points along the GT lanes.

        Args:
            map_gt_bboxes_3d: Lane bounding boxes for a single batch.

        Returns:
            Lane points of shape (L, bev_lane_points, 2).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance_points_list = []
        max_x, max_y = map_gt_bboxes_3d.max_x, map_gt_bboxes_3d.max_y

        for instance in map_gt_bboxes_3d.instance_list:
            # Sample points along the lane
            distances = np.linspace(0, instance.length, self.bev_lane_points)
            sampled_points = np.array([instance.interpolate(distance).coords for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)

        if not instance_points_list:
            return torch.empty(0, self.embed_dims, device=device)

        instance_points_array = np.array(instance_points_list)
        lane_pc_xy = to_tensor(instance_points_array).to(dtype=torch.float32, device=device)

        # Clamp to valid range
        lane_pc_xy[:, :, 0] = torch.clamp(lane_pc_xy[:, :, 0], min=-max_x, max=max_x)
        lane_pc_xy[:, :, 1] = torch.clamp(lane_pc_xy[:, :, 1], min=-max_y, max=max_y)

        return lane_pc_xy

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
