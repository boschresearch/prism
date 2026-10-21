# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional
from vlp.vlp.utils import gather_tensor
from vlp.core.registry import VLP_MODULES
from mmcv.utils import get_logger
from mmcv import ConfigDict

DataEncoderConfig = ConfigDict
FeatureAlignerConfig = ConfigDict

logger = get_logger("slp_modular")


@VLP_MODULES.register_module()
class SLP_Planner(nn.Module):
    """Generic Vision-Language Planning framework. SLP Part.
    Performs alignement between ego <-> agent and ego <-> lanes interaction queries with their corresponding target features.
    """

    def __init__(
        self,
        target_encoder: DataEncoderConfig,
        query_encoder: DataEncoderConfig,
        feature_aligner: FeatureAlignerConfig,
        contra_logit_scale_init: float = 4.6052,
        align_concatenated_input: bool = False,
        train_target_encoder_only: bool = False,
        **kwargs,
    ) -> None:
        """Initialize SLP Planner with flexible encoder architecture.

        Args:
            target_encoder: Encoder for target features (e.g., LLMEncoder, ImageEncoder, etc.)
            query_encoder: Wrapper to process ego-queries
            feature_aligner: Feature alignment module for computing losses
            pc_range: Point cloud range for bbox normalization
            embed_dims: Dimension of embedding space
            contra_logit_scale_init: Initial value for logit scaling parameter. Used for contrastive losses only.
            train_target_encoder_only: If True, only the target encoder is trained. All connection to original pipeline is detached.
            **kwargs: Additional arguments passed to encoders
        """
        super().__init__()
        self.align_concatenated_input = align_concatenated_input
        self.train_target_encoder_only = train_target_encoder_only

        # Initialize encoders
        self.target_encoder = VLP_MODULES.build(target_encoder)
        self.query_encoder = VLP_MODULES.build(query_encoder)

        # Feature alignment module
        feature_aligner["contra_logit_scale_init"] = contra_logit_scale_init
        self.feat_aligner = VLP_MODULES.build(feature_aligner)
        self.loss_class = self.feat_aligner.loss_class

        # Log trainable parameters
        logger.info(f"Target Encoder Trainable Params: \n  {self.get_trainable_params_list(self.target_encoder)}")
        logger.info(f"Source Encoder Trainable Params: \n {self.get_trainable_params_list(self.query_encoder)}")

    def get_trainable_params_list(self, module: nn.Module) -> List[str]:
        """Get a list of trainable parameters in the module."""
        return [name for name, param in module.named_parameters() if param.requires_grad]

    def get_target_feats(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract target features (e.g., text, image, etc.) from the input information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Ego agent features [B, embed_dims]
                - Ego lane features [B, 2 * embed_dims]
        """
        egoagent_query, egolane_query = self.target_encoder.get_features(**kwargs)
        return egoagent_query, egolane_query

    def get_query_features(
        self, egoagent_query: torch.Tensor, egolane_query: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from ego-agent and ego-lanes interaction queries.

        Args:
            egoagent_query: Ego agent query tensor [B, 1, embed_dims]
            egolane_query: Ego lane query tensor   [B, 1, embed_dims]
            **kwargs: Additional arguments for query encoder

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Processed ego agent query features [B, embed_dims]
            - Processed ego lane query features [B,  embed_dims]
        """
        egoagent_query, egolane_query = self.query_encoder.get_features(
            egoagent_query=egoagent_query, egolane_query=egolane_query, **kwargs
        )
        return egoagent_query, egolane_query

    def compute_slp_losses(
        self,
        egoagent_query: torch.Tensor,
        egolane_query: torch.Tensor,
        gt_bboxes_3d: List,
        ego_lcf_feat: Optional[torch.Tensor] = None,
        ego_fut_trajs: Optional[torch.Tensor] = None,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_labels_3d: Optional[List[torch.Tensor]] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_labels_3d: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        ego_his_trajs: Optional[torch.Tensor] = None,
        return_refined_queries: bool = False,
        sample_refined_queries_from_posterior: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute BEV alignment losses.

        Returns:
            Dictionary of BEV losses
        """
        output = {}
        slp_losses = dict()

        # Collect predicted queries (source)
        egoagent_query_predicted, egolane_query_predicted = self.get_query_features(
            egoagent_query=egoagent_query,
            egolane_query=egolane_query,
            gt_bboxes_3d=gt_bboxes_3d,
            ego_lcf_feat=ego_lcf_feat,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
        )

        if self.train_target_encoder_only:
            # remove gradient flow from original pipeline
            egoagent_query = egoagent_query.clone().detach()
            egolane_query = egolane_query.clone().detach()

        # Extract target features (target)
        egoagent_query_gt, egolane_query_gt = self.get_target_feats(
            egoagent_query=egoagent_query,
            egolane_query=egolane_query,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_attr_labels=gt_attr_labels,
            map_gt_labels_3d=map_gt_labels_3d,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            ego_his_trajs=ego_his_trajs,
            **kwargs,
        )

        if return_refined_queries:
            # return not aggregated queries (target only)
            output["refined_ego_agent_query"] = egoagent_query_gt

        if self.loss_class == "ContrastiveLoss":
            # Gather features across distributed processes
            egoagent_query_gt = gather_tensor(egoagent_query_gt, dim=0)  # [BG, D]
            egolane_query_gt = gather_tensor(egolane_query_gt, dim=0)
            egoagent_query_predicted = gather_tensor(egoagent_query_predicted, dim=0)
            egolane_query_predicted = gather_tensor(egolane_query_predicted, dim=0)

        # Remove extra dim
        if egoagent_query_gt.ndim == 3:
            egoagent_query_gt = egoagent_query_gt.squeeze(1)  # [BG, 1, D] -> [BG, D]
            egolane_query_gt = egolane_query_gt.squeeze(1)

        if self.align_concatenated_input:
            query_predicted = torch.cat([egoagent_query_predicted, egolane_query_predicted], dim=1)  # [BG, 2 * D]
            query_gt = torch.cat([egoagent_query_gt, egolane_query_gt], dim=1)  # [BG, 2 * D]

            if self.train_target_encoder_only:
                query_predicted = query_predicted.clone().detach()

            loss = self.feat_aligner(
                source=query_predicted,
                target=query_gt,
                loss_key_suffix="joint",
                return_posterior=sample_refined_queries_from_posterior,
            )

            posterior_agoagent = loss.pop("posterior", None)
            slp_losses.update(loss)

        else:
            # Calculate SLP loss
            if self.train_target_encoder_only:
                egoagent_query_predicted = egoagent_query_predicted.clone().detach()
                egolane_query_predicted = egolane_query_predicted.clone().detach()

            egoagent_loss = self.feat_aligner(
                source=egoagent_query_predicted,
                target=egoagent_query_gt,
                loss_key_suffix="slp_egoagent",
                return_posterior=sample_refined_queries_from_posterior,
            )

            egolane_loss = self.feat_aligner(
                source=egolane_query_predicted,
                target=egolane_query_gt,
                loss_key_suffix="slp_egolane",
                return_posterior=False,
            )

            posterior_agoagent = egoagent_loss.pop("posterior", None)
            slp_losses.update(egoagent_loss)
            slp_losses.update(egolane_loss)

        if return_refined_queries and posterior_agoagent is not None:
            # TODO: test on shape compatibility
            refined_ego_agent_query_from_posterior = posterior_agoagent.rsample(())
            assert refined_ego_agent_query_from_posterior.shape == egoagent_query_gt.shape
            output["refined_ego_agent_query_from_posterior"] = refined_ego_agent_query_from_posterior

        output["slp_losses"] = slp_losses

        return output

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device
