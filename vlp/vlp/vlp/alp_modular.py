# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional
from vlp.vlp.utils import aggregate_bev_feats, gather_tensor, set_requires_grad
from vlp.core.registry import VLP_MODULES
from mmcv import ConfigDict

DataEncoderConfig = ConfigDict
FeatureAlignerConfig = ConfigDict

from mmcv.utils import get_logger

logger = get_logger("alp_modular")


@VLP_MODULES.register_module()
class ALP_Planner(nn.Module):
    """Generic Vision-Language Planning framework. Only BEV-feature alignment is implemented (ALP).

    This framework supports any combination of encoders (text, image, audio, etc.)
    paired with BEV encoders for autonomous driving applications.
    """

    def __init__(
        self,
        target_encoder: DataEncoderConfig,
        bev_encoder: DataEncoderConfig,
        feature_aligner: FeatureAlignerConfig,
        logit_scale_init: float = 4.6052,
        mode: str = "ego_agents_lanes_aggregation",
        **kwargs,
    ) -> None:
        """Initialize VLP Planner with flexible encoder architecture.

        Args:
            target_encoder: Encoder for target features (e.g., LLMEncoder, ImageEncoder, etc.)
            source_encoder: Optional encoder for source features (defaults to BEVEncoder)
            feature_aligner: Feature alignment module for computing losses
            pc_range: Point cloud range for bbox normalization
            bev_h: Height of BEV representation
            bev_w: Width of BEV representation
            bev_lane_points: Number of lane points for BEV processing
            embed_dims: Dimension of embedding space
            logit_scale_init: Initial value for logit scaling parameter
            **kwargs: Additional arguments passed to encoders
        """
        super().__init__()

        self.mode = mode

        # Initialize encoders
        self.target_encoder = VLP_MODULES.build(target_encoder)
        self.bev_encoder = VLP_MODULES.build(bev_encoder)

        # Feature alignment module
        feature_aligner["contra_logit_scale_init"] = logit_scale_init
        self.bev_feat_aligner = VLP_MODULES.build(feature_aligner)

        # Log trainable parameters
        logger.info(f"Target Encoder Trainable Params: \n  {self.get_trainable_params_list(self.target_encoder)}")
        logger.info(f"Source Encoder Trainable Params: \n {self.get_trainable_params_list(self.bev_encoder)}")

    def get_trainable_params_list(self, module: nn.Module) -> List[str]:
        """Get a list of trainable parameters in the module."""
        return [name for name, param in module.named_parameters() if param.requires_grad]

    def get_target_feats(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract target features (e.g., text, image, etc.) for ego, agents, and lanes.

        Returns:
            Tuple containing:
                - Combined target features
                - Agent batch indices
                - Lane batch indices
        """
        return self.target_encoder.get_features(**kwargs)

    def get_bev_features(
        self,
        bev_embed: torch.Tensor,
        gt_bboxes_3d: List,
        ego_lcf_feat: Optional[torch.Tensor] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Extract BEV features for ego, agents, and lanes.

        Args:
            bev_embed: BEV feature embeddings [HW, B, C]
            gt_bboxes_3d: List of 3D bounding boxes for agents
            ego_lcf_feat: Optional ego vehicle state features
            map_gt_bboxes_3d: Optional lane/map bounding boxes

        Returns:
            List of feature tensors for each batch
        """
        return self.bev_encoder.get_features(
            bev_embed=bev_embed,
            gt_bboxes_3d=gt_bboxes_3d,
            ego_lcf_feat=ego_lcf_feat,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            **kwargs,
        )

    def compute_bev_losses(
        self,
        bev_embed: torch.Tensor,
        gt_bboxes_3d: List = None,
        ego_lcf_feat: Optional[torch.Tensor] = None,
        ego_fut_trajs: Optional[torch.Tensor] = None,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_labels_3d: Optional[List[torch.Tensor]] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_labels_3d: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        return_bev_feats: bool = False,
        sample_bev_from_posterior: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute BEV alignment losses.

        Returns:
            Dictionary of BEV losses
        """
        # Preprocess BEV features and target features
        bev_out = self._preprocess_bev_feats(
            bev_embed=bev_embed,
            gt_bboxes_3d=gt_bboxes_3d,
            ego_lcf_feat=ego_lcf_feat,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            gt_labels_3d=gt_labels_3d,
            gt_attr_labels=gt_attr_labels,
            map_gt_labels_3d=map_gt_labels_3d,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            return_not_aggregated_bev_feats=return_bev_feats,  # Return non-aggregated BEV features
            **kwargs,
        )

        # Calculate BEV loss
        bev_losses = self.bev_feat_aligner(
            source=bev_out["bev_source_aggregated"],
            target=bev_out["bev_target_aggregated"],
            return_posterior=sample_bev_from_posterior,
            loss_key_suffix=kwargs.get("loss_key_suffix", "alp"),
        )
        posterior = bev_losses.pop("posterior", None)

        output = {}
        output["bev_losses"] = bev_losses

        # If posterior is not None, sample from posterior distribution
        if return_bev_feats:
            # (losses, (source bev, target bev))
            output["bev_source"] = bev_out["bev_source"]
            output["bev_target"] = bev_out["bev_target"]

            if posterior is not None:
                # sample from posterior distribution
                # TODO: make more generic
                b, hw, c = bev_out["bev_target"].shape
                bev_target_from_posterior = posterior.rsample(()).reshape(b, hw, c)

                assert (
                    bev_target_from_posterior.shape == bev_out["bev_target"].shape
                ), "Shape mismatch after sampling from posterior distribution"
                output["bev_target_from_posterior"] = bev_target_from_posterior

        return output

    def _preprocess_bev_feats(
        self,
        bev_embed: torch.Tensor,
        gt_bboxes_3d: List = None,
        ego_lcf_feat: Optional[torch.Tensor] = None,
        ego_fut_trajs: Optional[torch.Tensor] = None,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_labels_3d: Optional[List[torch.Tensor]] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_labels_3d: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        return_not_aggregated_bev_feats: bool = False,  # returns not aggregated across devices bev features
        **kwargs,
    ) -> Dict:

        # Collect target BEV features
        out_target = self.get_target_feats(
            bev_embed=bev_embed,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_attr_labels=gt_attr_labels,
            map_gt_labels_3d=map_gt_labels_3d,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            pc_range=self.bev_encoder.pc_range if hasattr(self.bev_encoder, "pc_range") else None,
            **kwargs,
        )

        # Collect predicted bev features (source)
        out_source = self.get_bev_features(
            bev_embed=bev_embed,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_attr_labels=gt_attr_labels,
            ego_lcf_feat=ego_lcf_feat,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
        )

        if self.mode == "ego_agents_lanes_aggregation":
            # Aggregate and reshape BEV features: [B(1+M+L), 256]
            bev_source_aggregated, bev_target_aggregated = aggregate_bev_feats(
                bev_gt_feat=out_target["feats"],
                bev_bev_feat=out_source["feats"],
                agent_batch_idx=out_target["agent_batch_idx"],
                lane_batch_idx=out_target["lane_batch_idx"],
            )
        elif self.mode == "posterior":
            # Aggregate BEV features for posterior tasks
            bev_source_aggregated = gather_tensor(
                out_source["feats"],
                dim=0,
            )
            bev_target_aggregated = gather_tensor(
                out_target["feats"],
                dim=0,
            )

        outputs = {
            "bev_source_aggregated": bev_source_aggregated,
            "bev_target_aggregated": bev_target_aggregated,
        }

        if return_not_aggregated_bev_feats:
            outputs["bev_source"] = out_source["feats"]
            outputs["bev_target"] = out_target["feats"]

        return outputs

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device
