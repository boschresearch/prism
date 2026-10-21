# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from torch import Tensor

from vlp.core.registry import VLP_MODULES
from vlp.core.encoders.base import DataEncoder

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from vlp.core.registry import VLP_MODULES
from vlp.core.prediction_models.heads import FlexibleMLP
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import LiDARInstanceLines
from vlp.core.encoders.base import DataEncoder
from vlp.vlp.utils import to_tensor, split_tensor_by_sorted_batch_idx
import copy
from torch import nn


class ConvBlock(nn.Module):
    """A simple convolutional block with Conv2d, BatchNorm2d, and ReLU activation. Convs do not change the spatial dimensions."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvModel(nn.Module):
    """A simple convolutional model with multiple ConvBlocks."""

    def __init__(self, input_channel, output_channel, num_layers=3):
        super().__init__()
        layers = []
        out_channels_list = [input_channel] + [output_channel] * (num_layers - 1)

        in_channel = input_channel
        for i in range(num_layers):
            layers.append(ConvBlock(in_channel, out_channels_list[i]))
            in_channel = out_channels_list[i]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, input_channel, H, W]
        Returns:
            Output tensor of shape [B, output_channel, H, W]
        """
        return self.model(x)


class DeconvBlock(nn.Module):
    """A simple deconvolutional block with ConvTranspose2d, Conv2d, BatchNorm2d, and ReLU activation."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeconvUpsampler(nn.Module):
    """A deconvolutional upsampler that increases the spatial dimensions of the input tensor."""

    def __init__(self, input_size, output_size, hidden_dim=128, output_dim=256):
        """
        Args:
            input_size (int): starting spatial size (e.g., 2 for 2x2)
            output_size (int): final spatial size (e.g., 100 for 100x100)
        """
        super().__init__()

        current_size = input_size
        layers = []

        while current_size < output_size // 2:
            layers.append(DeconvBlock(hidden_dim, hidden_dim))
            current_size *= 2

        # Add the final layer to match the desired output dimension
        layers.append(DeconvBlock(hidden_dim, output_dim))

        # Interpolate to the exact output size
        if current_size != output_size:
            layers.append(nn.Upsample(size=(output_size, output_size), mode="bilinear", align_corners=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@VLP_MODULES.register_module()
class ConvRefineBEVEncoder(DataEncoder):
    """BEV features encoder for VLP tasks.Takes original BEV features, and refines them based on GT data provided (e.g. ego trajectories, agent trajectories, etc.)."""

    def __init__(
        self,
        embed_dims: int = 256,
        traj_ts: int = 6,
        bev_lane_points: int = 200,
        extra_modalities: Dict[str, bool] = {},
        **kwargs,
    ) -> None:
        super().__init__(embed_dims)

        self.traj_ts = traj_ts
        self.bev_lane_points = bev_lane_points
        self.hidden_dim = embed_dims // 2

        self.use_extra_modalities = True if True in extra_modalities.values() else False
        self.extra_modalities = extra_modalities

        if self.use_extra_modalities:
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

        self.input_dim = 2 + extra_input_dim  # 2 for x, y trajectory coords

        # Projection layers
        self.ego_proj = nn.Linear(self.traj_ts * self.input_dim, self.hidden_dim)
        self.agent_proj = nn.Linear(self.traj_ts * self.input_dim, self.hidden_dim)
        self.lane_proj = nn.Linear(self.bev_lane_points * self.input_dim, self.hidden_dim)
        self.merge_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 16)  # To get later [B, hidden_dim, 4x4]

        # Upsampler and refiner
        self.deconv_upsampler = DeconvUpsampler(
            input_size=4, output_size=100, hidden_dim=self.hidden_dim, output_dim=self.embed_dims  # (4x4)  # (100x100)
        )

        self.conv_refiner = ConvModel(
            input_channel=self.embed_dims * 2,  # After merge_proj
            output_channel=self.embed_dims,  # Final output dimension
            num_layers=3,  # Number of convolutional layers
        )

    def get_features(
        self,
        bev_embed: Tensor,
        ego_fut_trajs: torch.Tensor,
        gt_labels_3d: List[torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        **kwargs,
    ) -> Tensor:
        """Returns the input BEV features as is.

        Args:
            bev_embed: [B, HW, C] BEV features

        Returns:
            Tensor: [B, HW, C] BEV features
        """
        device = bev_embed.device
        B, HW, C = bev_embed.shape
        H = W = int(HW ** 0.5)

        # Bring all models to device
        self.ego_proj.to(device)
        self.agent_proj.to(device)
        self.lane_proj.to(device)
        self.merge_proj.to(device)
        self.deconv_upsampler.to(device)
        self.conv_refiner.to(device)

        # Prepare GT input data
        input = self.prepare_input(
            ego_fut_trajs=ego_fut_trajs,
            gt_labels_3d=gt_labels_3d,
            ego_fut_cmd=ego_fut_cmd,
            gt_attr_labels=gt_attr_labels,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
        )

        # Project and pool input GT data to same size - [B, hidden_dim] each.
        ego_feats, agent_feats, lane_feats = self.project_and_pool(input)

        # Project, reshape
        gt_feats = torch.cat([ego_feats, agent_feats, lane_feats], dim=-1)  # [B, 3 * hidden_dim]
        gt_feats = self.merge_proj(gt_feats)  # [B, 3 * hidden_dim] -> [B, 16 * hidden_dim]
        gt_feats = gt_feats.view(B, self.hidden_dim, 4, 4)

        # Upsample
        gt_feats = self.deconv_upsampler(gt_feats)  # [B, embed_dims, 100, 100]

        # Concat with original BEV feature
        bev_embed = bev_embed.view(B, H, W, C)
        bev_embed = bev_embed.permute(0, 3, 1, 2)
        feats_combined = torch.cat([bev_embed, gt_feats], dim=1)  # [B, embed_dims * 2, H, W]

        # Refine BEV features
        bev_embed_refined = self.conv_refiner(feats_combined)  # [B, embed_dims, H, W]
        bev_embed_refined = bev_embed_refined.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, embed_dims]

        assert bev_embed_refined.shape == (
            B,
            HW,
            self.embed_dims,
        ), f"Expected output shape [B, HW, embed_dims], got {bev_embed_refined.shape}."

        return {"feats": bev_embed_refined}

    def project_and_pool(self, input: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects and pools the input data (ego, agents, lanes) to the same size.

        Args:
            input: Dictionary containing:
                - ego_input: Tensor of shape [B, T, 2 + one_hot_dim] for ego vehicle.
                - agent_input: Tensor of shape [BM, T, 2 + one_hot_dim] for agents.
                - lane_input: Tensor of shape [BL, bev_lane_points, 2 + one_hot_dim] for lanes.

        Returns:
            Tuple containing:
                - ego_feats: Projected and pooled features for ego vehicle.
                - agent_feats: Projected and pooled features for agents.
                - lane_feats: Projected and pooled features for lanes.
        """
        device = input["ego_input"].device

        B = input["ego_input"].shape[0]
        BM = input["agent_input"].shape[0]  # Batch size for agents
        BL = input["lane_input"].shape[0]  # Batch size for lanes

        agent_batch_idx = input["agent_batch_idx"].to(device)  # [BM,]
        lane_batch_idx = input["lane_batch_idx"].to(device)  # [BL,]

        # Project ego input
        ego_input = input["ego_input"].reshape(B, -1)  # (B, T * (2 + one_hot_dim))
        ego_feats = self.ego_proj(ego_input)  # (B, hidden_dim)

        # Project agent input
        agent_input = input["agent_input"].reshape(BM, -1)  # (BM, T * (2 + one_hot_dim))
        agent_feats = self.agent_proj(agent_input)  # (BM, hidden_dim)

        # Project lane input
        lane_input = input["lane_input"].reshape(BL, -1)  # (BL, bev_lane_points * (2 + one_hot_dim))
        lane_feats = self.lane_proj(lane_input)  # (BL, hidden_dim)

        # Split agent and lane features by batch indices
        agent_feats = split_tensor_by_sorted_batch_idx(agent_feats, agent_batch_idx)  # B * [M, hidden_dim]
        lane_feats = split_tensor_by_sorted_batch_idx(lane_feats, lane_batch_idx)  # B * [L, hidden_dim]

        # Aggregate agent and lane features
        agent_feats = torch.stack([torch.mean(agent_feats[bi], dim=0) for bi in range(B)], dim=0)  # [B, hidden_dim]

        lane_feats = torch.stack([torch.mean(lane_feats[bi], dim=0) for bi in range(B)], dim=0)  # [B, hidden_dim]

        assert (
            ego_feats.shape == agent_feats.shape and agent_feats.shape == lane_feats.shape
        ), f"Expected all features to have the same shape, got ego: {ego_feats.shape}, agent: {agent_feats.shape}, lane: {lane_feats.shape}."

        return ego_feats, agent_feats, lane_feats

    def prepare_input(
        self,
        ego_fut_trajs: torch.Tensor,
        gt_labels_3d: List[torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
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

        B = ego_fut_trajs.shape[0]

        # Generate ego input (trajectories + one-hot encoding)
        ego_fut_trajs = ego_fut_trajs[:, 0, :, :]  # Shape: (B, T, 2)
        ego_input = self._merge_traj_with_one_hot(ego_fut_trajs, ego_one_hot)  # (B, T, 2 + one_hot_dim)

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
        else:
            agent_input = torch.empty(0, self.traj_ts, self.full_input_dim, device=device)

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
            else:
                lane_input = torch.empty(
                    0, self.bev_lane_points, self.full_input_dim, device=device
                )  # (BL++, bev_lane_points, 2 + one_hot_dim)

        return {
            "ego_input": ego_input,
            "agent_input": agent_input,
            "lane_input": lane_input,
            "agent_batch_idx": torch.tensor(agent_batch_idx),
            "lane_batch_idx": torch.tensor(lane_batch_idx),
        }

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
