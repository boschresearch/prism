# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import copy
import numpy as np
import cv2
import torch
from typing import List, Optional, Dict, Any

from vlp.core.registry import VLP_MODULES
from vlp.core.encoders.base import DataEncoder
from vlp.vlp.utils import to_tensor


@VLP_MODULES.register_module()
class BEVEncoder(DataEncoder):
    """Process BEV features for VLP tasks."""

    def __init__(
        self,
        embed_dims: int = 256,
        pc_range: List[float] = None,
        bev_h: int = 100,
        bev_w: int = 100,
        bev_lane_points: int = 200,
    ):
        super().__init__(embed_dims)
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_lane_points = bev_lane_points

    def get_features(
        self,
        bev_embed: torch.Tensor,
        gt_bboxes_3d: List,
        ego_lcf_feat: Optional[torch.Tensor] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract BEV features for ego, agents, and lanes.

        Args:
            bev_embed: [HW, B, C] BEV features
            gt_bboxes_3d: B x Lidar.tensor[Mb, 9]
            ego_lcf_feat: [B, 1, 1, 9] ego state features
            map_gt_bboxes_3d: B x Line[Lb, ...] lane features

        Returns:
            Dict[str, Any]:  For key 'feats':  List[torch.Tensor] with shape B x [(1 + M++ + L++), embed_dims] features per batch,
                                (1 - ego, M - agents, L - lanes, ++ - varying size)
        """
        bev_embed = bev_embed.permute(1, 0, 2)  # [B, HW, C]
        B, HW, C = bev_embed.shape
        bev_embed = bev_embed.view(B, self.bev_h, self.bev_w, C)

        batch_features = []

        for i in range(B):
            # Extract ego features
            ego_feat = self._extract_ego_features(bev_embed[i], ego_lcf_feat[i] if ego_lcf_feat is not None else None)

            # Extract agent features
            agent_feats = self._extract_agent_features(bev_embed[i], gt_bboxes_3d[i])

            # Extract lane features if available
            if map_gt_bboxes_3d:
                lane_feats = self._extract_lane_features(bev_embed[i], map_gt_bboxes_3d[i])
                combined_feats = torch.cat([ego_feat.unsqueeze(0), agent_feats, lane_feats], dim=0)
            else:
                combined_feats = torch.cat([ego_feat.unsqueeze(0), agent_feats], dim=0)

            batch_features.append(combined_feats)

        return {"feats": batch_features}

    def _extract_ego_features(
        self, bev_feat: torch.Tensor, ego_lcf_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract ego vehicle features from BEV.

        Args:
            bev_feat: [H, W, C]
            ego_lcf_feat: [1, 1, 9], 9 = (vx, vy, ax, ay, w, length, width, vel, steer), w = yaw

        Returns:
            torch.Tensor of shape [C,] -> aggregated ego features for single BEV
        """
        if ego_lcf_feat is None:
            raise ValueError("Not Tested Yet")
            # Default to center region if no ego info available
            h_center = self.bev_h // 2
            w_center = self.bev_w // 2
            ego_region = bev_feat[h_center - 2 : h_center + 3, w_center - 2 : w_center + 3, :]
            return ego_region.flatten(0, 1).mean(0)

        # Use ego vehicle dimensions to extract features
        sdc_pc_front = ego_lcf_feat[0, 0, 5] / 2
        sdc_pc_back = -ego_lcf_feat[0, 0, 5] / 2
        sdc_pc_right = ego_lcf_feat[0, 0, 6] / 2
        sdc_pc_left = -ego_lcf_feat[0, 0, 6] / 2

        pc_h = self.pc_range[4] - self.pc_range[1]
        pc_w = self.pc_range[3] - self.pc_range[0]

        sdc_ah_front = (self.bev_h * (self.pc_range[4] - sdc_pc_front) / pc_h).int()
        sdc_ah_back = (self.bev_h * (self.pc_range[4] - sdc_pc_back) / pc_h).int()
        sdc_aw_right = (self.bev_w * (sdc_pc_right - self.pc_range[0]) / pc_w).int()
        sdc_aw_left = (self.bev_w * (sdc_pc_left - self.pc_range[0]) / pc_w).int()

        sdc_ah_front = torch.clamp(sdc_ah_front, min=0, max=self.bev_h - 1)
        sdc_ah_back = torch.clamp(sdc_ah_back, min=0, max=self.bev_h - 1)
        sdc_aw_right = torch.clamp(sdc_aw_right, min=0, max=self.bev_w - 1)
        sdc_aw_left = torch.clamp(sdc_aw_left, min=0, max=self.bev_w - 1)

        ego_region = bev_feat[sdc_ah_front : sdc_ah_back + 1, sdc_aw_left : sdc_aw_right + 1, :]
        return ego_region.flatten(0, 1).mean(0)  # [c,]

    def _extract_agent_features(self, bev_feat: torch.Tensor, gt_bboxes_3d) -> torch.Tensor:
        """Extract agent features from BEV using bounding boxes

        Args:
            bev_feat: [H, W, C]
            gt_bboxes_3d:

        Returns:
            torch.Tensor: [M, C], where M - number of agents

        """
        device = bev_feat.device
        agent_pc_xy = copy.deepcopy(gt_bboxes_3d.corners[:, ::2, :2])  # [M, 4, xy]

        pc_h = self.pc_range[4] - self.pc_range[1]
        pc_w = self.pc_range[3] - self.pc_range[0]

        ### convert agent 3d bbox corners to BEV pixel coords ###
        agent_array_y = (self.bev_h * (self.pc_range[4] - agent_pc_xy[:, :, 1]) / pc_h).int()
        agent_array_x = (self.bev_w * (agent_pc_xy[:, :, 0] - self.pc_range[0]) / pc_w).int()
        agent_array_xy = torch.stack([agent_array_x, agent_array_y], dim=-1)  # [M, 4, xy]

        M = agent_array_xy.shape[0]  # number of agents
        agent_array_xy = agent_array_xy.detach().cpu().numpy().astype(np.int32)

        agent_features = []  # M x [C,]
        for ai in range(M):
            # fill mask of BEV features size, where 2d bbox of agent is 1 and the rest is 0
            mask = np.zeros([self.bev_h, self.bev_w], dtype=np.int32)
            mask = cv2.fillConvexPoly(
                mask,
                np.array(
                    [agent_array_xy[ai, 0], agent_array_xy[ai, 1], agent_array_xy[ai, 3], agent_array_xy[ai, 2]],
                    dtype=np.int32,
                ),
                color=(1,),
            )
            mask = torch.tensor(mask).to(torch.bool).to(device).unsqueeze(-1)  # [h, w, 1]

            masked_feat = mask * bev_feat  # [h, w, c]
            agent_feat = masked_feat.sum(dim=(0, 1)) / mask.sum()  # [c,]
            agent_features.append(agent_feat)

        return torch.stack(agent_features, dim=0)

    def _extract_lane_features(self, bev_feat: torch.Tensor, map_gt_bboxes_3d) -> torch.Tensor:
        """Extract lane features from BEV using lane points

        Args:
            bev_feat: [H, W, C], for B = 1
            map_gt_bboxes_3d: instance of BaseInstance3DBoxes. For B = 1

        Returns:
            torch.Tensor: [L, C], where L - number of lanes
        """
        device = bev_feat.device

        # Sample points along lanes
        instance_points_list = []
        max_x, max_y = map_gt_bboxes_3d.max_x, map_gt_bboxes_3d.max_y

        for instance in map_gt_bboxes_3d.instance_list:
            # sample points along the lane
            distances = np.linspace(0, instance.length, self.bev_lane_points)
            sampled_points = np.array([instance.interpolate(distance).coords for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)

        if not instance_points_list:
            return torch.empty(0, bev_feat.shape[-1], device=device)

        instance_points_array = np.array(instance_points_list)
        lane_pc_xy = to_tensor(instance_points_array).to(dtype=torch.float32)

        # Clamp to valid range
        lane_pc_xy[:, :, 0] = torch.clamp(lane_pc_xy[:, :, 0], min=-max_x, max=max_x)
        lane_pc_xy[:, :, 1] = torch.clamp(lane_pc_xy[:, :, 1], min=-max_y, max=max_y)

        # Convert to pixel coordinates
        pc_h = self.pc_range[4] - self.pc_range[1]
        pc_w = self.pc_range[3] - self.pc_range[0]

        lane_array_y = (self.bev_h * (self.pc_range[4] - lane_pc_xy[:, :, 1]) / pc_h).int().long()
        lane_array_x = (self.bev_w * (lane_pc_xy[:, :, 0] - self.pc_range[0]) / pc_w).int().long()

        lane_array_y = torch.clamp(lane_array_y, min=0, max=self.bev_h - 1)
        lane_array_x = torch.clamp(lane_array_x, min=0, max=self.bev_w - 1)

        L, pts = lane_array_x.shape
        lane_features = []

        # NOTE: here was bug in original implementation! - only extract from 0-th batch
        """"
         L, pts = lane_array_x.shape
        batch_idx = torch.zeros(
            [
                pts,
            ],
            dtype=torch.long,
        )
        lane_bev = []
        for ai in range(L):
            # extract features at the lane points from previously extracted BEV features
            cur_li_bev = bev_embed[
                batch_idx,
                lane_array_y[ai],
                lane_array_x[ai],
            ]  # [20, C]
            cur_li_bev = cur_li_bev.mean(0)  # [c,]
            lane_bev.append(cur_li_bev)
        lane_bev = torch.stack(lane_bev, dim=0)  # [L, C]
        """
        for ai in range(L):
            lane_feat = bev_feat[lane_array_y[ai], lane_array_x[ai], :].mean(0)  # [20, C] -> [C,]
            lane_features.append(lane_feat)

        return torch.stack(lane_features, dim=0)  # [L, C]
