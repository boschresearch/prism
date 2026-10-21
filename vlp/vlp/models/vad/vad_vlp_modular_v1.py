# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

### VLP PLUGIN ### - Original VLP implementation but with modular loss definition
import os
import time
import copy

import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric

###################################################################
from vlp.vlp.vlp import VLP_Planner
from vlp.core.registry import VLP_LOSSES, VLP_MODULES
from vlp.vlp.utils import aggregate_bev_feats

###################################################################
from mmcv.utils import get_logger
from mmcv.utils import build_from_cfg
from collections import OrderedDict
import torch.distributed as dist
from pathlib import Path
import mmcv

logger = get_logger("VAD_VLP_Prob")


def init_model_from_cktp(ckpt: str, model: torch.nn.Module, module_name: str = None):
    state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]

    if module_name is not None:
        # filter state_dict to only include keys that start with module_name
        state_dict = {k[len(module_name) + 1 :]: v for k, v in state_dict.items() if k.startswith(module_name)}
        logger.info(f"Filtered state dict to only include keys: {state_dict.keys()}.")

    model.load_state_dict(state_dict, strict=False)

    return model


@DETECTORS.register_module()
class VAD_VLP_Probabilistic(MVXTwoStageDetector):
    """VAD model using the VLP package."""

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        fut_ts=6,
        fut_mode=6,
        bev_prob_encoder=None,
        inter_query_prob_encoder=None,
        alp_cfg=None,
        slp_cfg=None,
        use_posterior_inference=False,
        init_prior_slp_encoder_from=None,
    ):
        """VAD_VLP model for VLP tasks.

        Args:
            alp_posterior_training_cfg (dict, optional): Config for posterior inference of ALP module.
            Defines whether to use externally refined BEV features for final model inference. Used during training only. Keys: apply, rantom_rate.

        """
        super(VAD_VLP_Probabilistic, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head["valid_fut_ts"]

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.planning_metric = None

        # Initialize ALP module
        if alp_cfg is not None:
            logger.info(f"Initialize ALP module, class: {alp_cfg['type']}.")
            self.alp = build_from_cfg(alp_cfg, VLP_MODULES)
        else:
            logger.warning("ALP module is not specified.")
            self.alp = None

        # Initialize SLP module
        if slp_cfg is not None:
            logger.info(f"Initialize SLP module, class: {slp_cfg['type']}.")
            self.slp = build_from_cfg(slp_cfg, VLP_MODULES)
        else:
            logger.warning("SLP module is not specified.")
            self.slp = None

        # Intialize Probability Encoders
        if bev_prob_encoder is not None:
            self.bev_prob_encoder = build_from_cfg(bev_prob_encoder, VLP_MODULES)
            logger.info(f"Initialized BEV Probabilistic Encoder, class: {bev_prob_encoder['type']}.")
            logger.info(
                f"[BEV Prob Encoder] Trainable Params: \n  {self.get_trainable_params_list(self.bev_prob_encoder)}"
            )
        else:
            self.bev_prob_encoder = None
            logger.warning("BEV Probabilistic Encoder is not specified.")

        if inter_query_prob_encoder is not None:
            self.egoagent_query_prob_encoder = build_from_cfg(inter_query_prob_encoder, VLP_MODULES)

            if init_prior_slp_encoder_from is not None:
                logger.info(f"Initialize Inter Query Prob Encoder from: {init_prior_slp_encoder_from}")
                self.egoagent_query_prob_encoder = init_model_from_cktp(
                    init_prior_slp_encoder_from,
                    self.egoagent_query_prob_encoder,
                    module_name="egoagent_query_prob_encoder",
                )

                # reinitialize logvar_lin_layers if not none
                if self.egoagent_query_prob_encoder.logvar_layer is not None:
                    self.egoagent_query_prob_encoder.logvar_layer.init_weights(final="zeros", zero_bias=False)
                    logger.info(f"Reinitialized logvar_layer of Inter Query Prob Encoder.")

            logger.info(f"Initialized Inter Query Probabilistic Encoder, class: {inter_query_prob_encoder['type']}.")
            logger.info(
                f"[Inter Query Prob Encoder] Trainable Params: \n  {self.get_trainable_params_list(self.egoagent_query_prob_encoder)}"
            )
        else:
            self.egoagent_query_prob_encoder = None
            logger.warning("Inter Query Probabilistic Encoder is not specified.")

        self.use_posterior_inference = use_posterior_inference

    def get_trainable_params_list(self, module):
        """Get a list of trainable parameters in the module."""
        return [name for name, param in module.named_parameters() if param.requires_grad]

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        map_gt_bboxes_ignore=None,
        prev_bev=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_masks=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        # Initialize VLP losses
        vlp_losses = dict()
        alp_losses = {}
        slp_losses = {}

        # Get BEV features
        bev_source = self.pts_bbox_head(pts_feats, img_metas, prev_bev, only_bev=True)

        if self.bev_prob_encoder is not None:
            raise NotImplementedError("Probabilistic BEV Encoder is not implemented yet. ")
            bev_source_dist = self.bev_prob_encoder.get_distribution(bev_source)

        if self.alp is not None:
            raise NotImplementedError("Not implemented yet.")

            bev_target_dist = self.alp.get_target_dist(
                bev_source_dist=bev_source_dist,
                ego_lcf_feat=ego_lcf_feat,
                ego_fut_trajs=ego_fut_trajs,
                ego_fut_cmd=ego_fut_cmd,
                gt_labels_3d=gt_labels_3d,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_attr_labels=gt_attr_labels,
                map_gt_labels_3d=map_gt_labels_3d,
                map_gt_bboxes_3d=map_gt_bboxes_3d,
            )

            alp_losses = self.alp.compute_bev_loss(bev_source_dist=bev_source_dist, bev_target_dist=bev_target_dist)

            if self.use_posterior_inference:
                bev_source = bev_target_dist.rsample()
            else:
                bev_source = bev_source_dist.rsample()

        elif self.bev_prob_encoder is not None:
            raise NotImplementedError("not implemented yet.")
            bev_source = bev_source_dist.rsample()

        # Get interaction queries
        queries = self.pts_bbox_head(
            pts_feats,
            img_metas,
            prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
            external_bev_features=bev_source,
            only_ego_queries=True,
        )

        ego_agent_query = queries["ego_agent_query"].squeeze(1)  # [B, 1, D] -> [B, D]
        ego_lane_query = queries["ego_map_query"].squeeze(1)

        if self.egoagent_query_prob_encoder is not None:
            egoagent_query_dist = self.egoagent_query_prob_encoder.get_distribution(
                ego_agent_query, ego_fut_cmd=ego_fut_cmd
            )

        if self.slp is not None:
            egoagent_query_target_dist = self.slp.get_target_dist(
                query_source_dist=egoagent_query_dist,
                ego_agent_query=ego_agent_query,
                gt_bboxes_3d=gt_bboxes_3d,
                ego_lcf_feat=ego_lcf_feat,
                ego_fut_trajs=ego_fut_trajs,
                ego_fut_cmd=ego_fut_cmd,
                gt_labels_3d=gt_labels_3d,
                gt_attr_labels=gt_attr_labels,
                map_gt_labels_3d=map_gt_labels_3d,
                map_gt_bboxes_3d=map_gt_bboxes_3d,
                ego_his_trajs=ego_his_trajs,
            )

            egoagent_query_loss = self.slp.compute_query_loss(
                query_target_dist=egoagent_query_target_dist, query_source_dist=egoagent_query_dist, suffix="ageagent"
            )

            slp_losses.update(egoagent_query_loss)

            if self.use_posterior_inference:
                ego_agent_query = egoagent_query_target_dist.rsample()
            else:
                ego_agent_query = egoagent_query_dist.rsample()

        elif self.egoagent_query_prob_encoder is not None:
            ego_agent_query = egoagent_query_dist.rsample()

        ############################ Final Model Inference ###################################
        inference_kwargs = {
            "mlvl_feats": pts_feats,
            "img_metas": img_metas,
            "prev_bev": prev_bev,
            "ego_his_trajs": ego_his_trajs,
            "ego_lcf_feat": ego_lcf_feat,
            "external_bev_features": bev_source,
            "external_ego_agent_query": ego_agent_query,
            "ego_fut_cmd": ego_fut_cmd,
        }

        # Final inference with all refinements
        outs = self.pts_bbox_head(**inference_kwargs)

        ############################ Update Final Losses ###################################
        if self.alp is not None:
            vlp_losses.update(alp_losses)

        if self.slp is not None:
            vlp_losses.update(slp_losses)
        ###################################################################################################

        # Compute all model's losses
        loss_inputs = [
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            outs,
            ego_fut_trajs,
            ego_fut_masks,
            ego_fut_cmd,
            gt_attr_labels,
        ]

        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        return losses, vlp_losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            if kwargs.pop("eval_posterior", False):
                return self.forward_test_posterior(**kwargs)
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=("img", "points", "prev_bev"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        map_gt_bboxes_3d=None,
        map_gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        map_gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_masks=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # print('forward_trai img shape: ', img.shape, type(img)) #[B=1, T=3, V=6, C=3, H=384, W=640] #<class 'torch.Tensor'>
        # import pdb; pdb.set_trace()

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts, vlp_loss = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            map_gt_bboxes_ignore,
            prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_masks=ego_fut_masks,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )

        ###############    VLP loss update    ############
        losses.update(vlp_loss)
        ##################################################
        losses.update(losses_pts)
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        sample_n_ego_trajs=1,
        save_slp_query_path=None,
        **kwargs,
    ):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            sample_n_ego_trajs=sample_n_ego_trajs,
            save_slp_query_path=save_slp_query_path,
            **kwargs,
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        sample_n_ego_trajs=1,
        save_slp_query_path=None,
        **kwargs,
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            sample_n_ego_trajs=sample_n_ego_trajs,
            save_slp_query_path=save_slp_query_path,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = metric_dict

        if save_slp_query_path is not None:
            # save images
            image_path = Path(save_slp_query_path) / "images"
            image_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "img": img.cpu(),
                    "img_metas": img_metas,
                },
                image_path / f"{img_metas[0]['sample_idx']}_img.pth",
            )

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        sample_n_ego_trajs=1,
        save_slp_query_path=None,
    ):
        """Test function"""
        mapped_class_names = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]

        # ------------------------ (1) FIRST PASS: full run ------------------------
        # Get BEV features
        bev_source = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, only_bev=True)

        if self.bev_prob_encoder is not None:
            raise NotImplementedError("Probabilistic BEV Encoder is not implemented yet.")
            # bev_source_dist = self.bev_prob_encoder.get_distribution(bev_source)
            # bev_source = bev_source_dist.rsample()

        # Get interaction queries
        queries = self.pts_bbox_head(
            x,
            img_metas,
            prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
            external_bev_features=bev_source,
            only_ego_queries=True,
        )
        ego_agent_query = queries["ego_agent_query"].squeeze(1)
        ego_lane_query = queries["ego_map_query"].squeeze(1)

        if self.egoagent_query_prob_encoder is not None:
            egoagent_query_dist = self.egoagent_query_prob_encoder.get_distribution(
                ego_agent_query, ego_fut_cmd=ego_fut_cmd
            )
            ego_agent_query = egoagent_query_dist.rsample()

        outs = self.pts_bbox_head(
            x,
            img_metas,
            prev_bev=prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
            external_bev_features=bev_source,
            external_ego_agent_query=ego_agent_query,
            ego_fut_cmd=ego_fut_cmd,
        )

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result["trajs_3d"] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i].cpu()
            bbox_result["ego_fut_cmd"] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, "only support batch_size=1 now"
        score_threshold = 0.6
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)
            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to("cpu")
            gt_attr_label = gt_attr_labels[0][0].to("cpu")
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter preds by score
            mask = bbox_result["scores_3d"] > score_threshold
            bbox_result["boxes_3d"] = bbox_result["boxes_3d"][mask]
            bbox_result["scores_3d"] = bbox_result["scores_3d"][mask]
            bbox_result["labels_3d"] = bbox_result["labels_3d"][mask]
            bbox_result["trajs_3d"] = bbox_result["trajs_3d"][mask]

            # ---- full metrics only for the first pass
            matched_bbox_result = self.assign_pred_to_gt_vip3d(bbox_result, gt_bbox, gt_label)
            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result, matched_bbox_result, mapped_class_names
            )

            # ego planning metric (first pass only)
            assert ego_fut_trajs.shape[0] == 1, "only support batch_size=1 for testing"
            ego_fut_preds = bbox_result["ego_fut_preds"]  # [num_cmd, T, 2] (displacements)
            ego_fut_trajs_gt = ego_fut_trajs[0, 0]  # [T, 2] (displacements)
            ego_fut_cmd_ = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd_)[0, 0]
            ego_fut_pred_sel = ego_fut_preds[ego_fut_cmd_idx]  # [T, 2] (displacements)
            ego_fut_pred_sel = ego_fut_pred_sel.cumsum(dim=-2)  # to absolute traj
            ego_fut_trajs_gt = ego_fut_trajs_gt.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs=ego_fut_pred_sel[None],
                gt_ego_fut_trajs=ego_fut_trajs_gt[None],
                gt_agent_boxes=gt_bbox,
                gt_agent_feats=gt_attr_label.unsqueeze(0),
                fut_valid_flag=fut_valid_flag,
            )
            metric_dict.update(metric_dict_planner_stp3)

        # save queries
        if save_slp_query_path is not None and self.slp is not None:
            save_path = Path(save_slp_query_path) / "slp_egoagent_query_prior"
            save_path.mkdir(parents=True, exist_ok=True)
            save_dict = {
                "ego_fut_cmd_idx": ego_fut_cmd_idx.cpu(),
                "ego_agent_query": [ego_agent_query.cpu()],
            }

        # ------------------------ (2) EXTRA SAMPLES: ego traj only, cheap path ------------------------
        # We do NOT compute detection/map metrics again.
        if sample_n_ego_trajs > 1:
            extra_sampled_ego_trajs = []

            with torch.no_grad():
                for sample_id in range(sample_n_ego_trajs - 1):
                    # Resample BEV and queries (stochastic encoders)

                    # take original bev and query first
                    bev_source_s = bev_source
                    ego_agent_query_s = ego_agent_query

                    if self.bev_prob_encoder is not None:
                        # When implemented, uncomment:
                        # bev_source_s = bev_source_dist.rsample()
                        raise NotImplementedError("Probabilistic BEV Encoder is not implemented yet.")

                    if self.egoagent_query_prob_encoder is not None:
                        # sample new query from the distribution
                        ego_agent_query_s = egoagent_query_dist.rsample()

                    # Run a light forward that produces ego futures; we skip bbox postproc & metrics
                    outs_s = self.pts_bbox_head(
                        x,
                        img_metas,
                        prev_bev=prev_bev,
                        ego_his_trajs=ego_his_trajs,
                        ego_lcf_feat=ego_lcf_feat,
                        external_bev_features=bev_source_s,
                        external_ego_agent_query=ego_agent_query_s,
                    )

                    # Extract only the ego future trajectory for the selected cmd, convert to absolute
                    ego_fut_preds_s = outs_s["ego_fut_preds"][
                        0
                    ].cpu()  # [1, num_cmd, T, 2] -> [num_cmd, T, 2] displacements
                    extra_sampled_ego_trajs.append(ego_fut_preds_s)

                    if save_slp_query_path is not None and self.slp is not None:
                        save_dict["ego_agent_query"].append(ego_agent_query_s.cpu())

            # Attach to the first (and only) item’s results
            bbox_results[0]["extra_sampled_ego_fut_preds"] = extra_sampled_ego_trajs

        if save_slp_query_path is not None and self.slp is not None:
            torch.save(save_dict, save_path / f"{img_metas[0]['sample_idx']}_feats.pth")

        return outs["bev_embed"], bbox_results, metric_dict

    def forward_test_posterior(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        save_slp_query_path=None,
        fut_valid_flag=None,  # NEW: optional, passed through to posterior call
        **kwargs,
    ):
        # input checks/packing (same as forward_test)
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        # scene boundary handling
        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # can-bus delta handling
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        # --- posterior pass ---
        new_prev_bev, pts_bbox, metric_dict = self.forward_with_posterior(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            save_slp_query_path=save_slp_query_path,
            fut_valid_flag=fut_valid_flag,
            **kwargs,
        )

        # Build the same outer structure as simple_test
        bbox_list = [dict() for _ in range(len(pts_bbox))]
        for result_dict, pts_bbox_item in zip(bbox_list, pts_bbox):
            result_dict["pts_bbox"] = pts_bbox_item
            result_dict["metric_results"] = metric_dict

        # Save images (mirror simple_test) if requested
        if save_slp_query_path is not None:
            from pathlib import Path

            image_path = Path(save_slp_query_path) / "images"
            image_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "img": img[0].cpu(),
                    "img_metas": img_metas[0],
                },
                image_path / f"{img_metas[0][0]['sample_idx']}_img.pth",
            )

        # state update
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        # match forward_test return (list of dicts)
        return bbox_list

    def forward_with_posterior(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        save_slp_query_path=None,
        fut_valid_flag=None,  # NEW: optional, mirrors simple_test_pts
        **kwargs,
    ):
        """
        Posterior-driven evaluation (single pass). Returns:
            new_prev_bev, pts_bbox_list, metric_dict
        """
        # -------------- Feature extraction --------------
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # -------------- Build prior BEV --------------
        bev_prior = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)

        # -------------- (Optional) posterior BEV via ALP --------------
        bev_used = bev_prior
        if (self.bev_prob_encoder is not None) and (self.alp is not None):
            bev_source_dist = self.bev_prob_encoder.get_distribution(bev_prior)
            bev_target_dist = self.alp.get_target_dist(
                bev_source_dist=bev_source_dist,
                ego_lcf_feat=ego_lcf_feat,
                ego_fut_trajs=ego_fut_trajs,
                ego_fut_cmd=ego_fut_cmd,
                gt_labels_3d=gt_labels_3d,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_attr_labels=gt_attr_labels,
                map_gt_labels_3d=kwargs.get("map_gt_labels_3d", None),
                map_gt_bboxes_3d=kwargs.get("map_gt_bboxes_3d", None),
            )
            bev_used = bev_target_dist.rsample()

        # -------------- Build prior ego-agent query --------------
        queries = self.pts_bbox_head(
            img_feats,
            img_metas,
            prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
            external_bev_features=bev_used,
            only_ego_queries=True,
        )
        ego_agent_query_prior = queries["ego_agent_query"].squeeze(1)

        # -------------- (Optional) posterior ego-agent query via SLP --------------
        ego_agent_query_used = ego_agent_query_prior
        if (self.egoagent_query_prob_encoder is not None) and (self.slp is not None):
            egoagent_query_source_dist = self.egoagent_query_prob_encoder.get_distribution(ego_agent_query_prior)
            egoagent_query_target_dist = self.slp.get_target_dist(
                query_source_dist=egoagent_query_source_dist,
                ego_agent_query=ego_agent_query_prior,
                gt_bboxes_3d=gt_bboxes_3d,
                ego_lcf_feat=ego_lcf_feat,
                ego_fut_trajs=ego_fut_trajs,
                ego_fut_cmd=ego_fut_cmd,
                gt_labels_3d=gt_labels_3d,
                gt_attr_labels=gt_attr_labels,
                map_gt_labels_3d=kwargs.get("map_gt_labels_3d", None),
                map_gt_bboxes_3d=kwargs.get("map_gt_bboxes_3d", None),
                ego_his_trajs=ego_his_trajs,
            )
            ego_agent_query_used = egoagent_query_target_dist.rsample()

        # -------------- Final forward (single pass) --------------
        outs = self.pts_bbox_head(
            img_feats,
            img_metas,
            prev_bev=prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
            external_bev_features=bev_used,
            external_ego_agent_query=ego_agent_query_used,
        )

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        # Build pts_bbox-style dicts (like simple_test_pts)
        pts_bbox_results = []
        for bboxes, scores, labels, trajs, map_bboxes, map_scores, map_labels, map_pts in bbox_list:
            r = bbox3d2result(bboxes, scores, labels)
            r["trajs_3d"] = trajs.cpu()
            r.update(self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts))
            r["ego_fut_preds"] = outs["ego_fut_preds"][0].cpu()
            r["ego_fut_cmd"] = ego_fut_cmd.cpu()
            pts_bbox_results.append(r)

        # -------- Metrics (mirror simple_test_pts) --------
        mapped_class_names = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]
        score_threshold = 0.6
        with torch.no_grad():
            assert len(pts_bbox_results) == 1, "only support batch_size=1 now"
            c_bbox_results = copy.deepcopy(pts_bbox_results)
            bbox_result = c_bbox_results[0]

            # filter preds by score
            mask = bbox_result["scores_3d"] > score_threshold
            bbox_result["boxes_3d"] = bbox_result["boxes_3d"][mask]
            bbox_result["scores_3d"] = bbox_result["scores_3d"][mask]
            bbox_result["labels_3d"] = bbox_result["labels_3d"][mask]
            bbox_result["trajs_3d"] = bbox_result["trajs_3d"][mask]

            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to("cpu")
            gt_attr_label = gt_attr_labels[0][0].to("cpu")
            fut_valid = bool(fut_valid_flag[0][0]) if fut_valid_flag is not None else False

            matched_bbox_result = self.assign_pred_to_gt_vip3d(bbox_result, gt_bbox, gt_label)
            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result, matched_bbox_result, mapped_class_names
            )

            # planner metric (same as simple_test_pts)
            assert ego_fut_trajs.shape[0] == 1, "only support batch_size=1 for testing"
            ego_fut_preds = bbox_result["ego_fut_preds"]  # [num_cmd, T, 2] displacements
            ego_fut_trajs_gt = ego_fut_trajs[0, 0]  # [T, 2] displacements
            ego_fut_cmd_ = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd_)[0, 0]
            ego_fut_pred_sel = ego_fut_preds[ego_fut_cmd_idx].cumsum(dim=-2)
            ego_fut_trajs_gt = ego_fut_trajs_gt.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs=ego_fut_pred_sel[None],
                gt_ego_fut_trajs=ego_fut_trajs_gt[None],
                gt_agent_boxes=gt_bbox,
                gt_agent_feats=gt_attr_label.unsqueeze(0),
                fut_valid_flag=fut_valid,
            )
            metric_dict.update(metric_dict_planner_stp3)

        # Optional: save posterior queries (if SLP present)
        if save_slp_query_path is not None and self.slp is not None:
            from pathlib import Path

            save_path = Path(save_slp_query_path) / "slp_egoagent_query_posterior"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "ego_agent_query_posterior": ego_agent_query_used.detach().cpu(),
                    "ego_agent_query_prior": ego_agent_query_prior.detach().cpu(),
                },
                save_path / f"{img_metas[0]['sample_idx']}_feats.pth",
            )

        return outs["bev_embed"], pts_bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to("cpu"),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to("cpu"),
        )

        if attrs is not None:
            result_dict["map_attrs_3d"] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(self, bbox_result, gt_bbox, gt_label, match_dis_thresh=2.0):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """
        dynamic_list = [0, 1, 3, 4, 6, 7, 8]
        matched_bbox_result = torch.ones((len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result["boxes_3d"].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result["labels_3d"]]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ["car", "pedestrian"]
        motion_metric_names = ["gt", "cnt_ade", "cnt_fde", "hit", "fp", "ADE", "FDE", "MR"]

        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met + "_" + cls] = 0.0

        veh_list = [0, 1, 3, 4]
        ignore_list = ["construction_vehicle", "barrier", "traffic_cone", "motorcycle", "bicycle"]

        for i in range(pred_bbox["labels_3d"].shape[0]):
            pred_bbox["labels_3d"][i] = 0 if pred_bbox["labels_3d"][i] in veh_list else pred_bbox["labels_3d"][i]
            box_name = mapped_class_names[pred_bbox["labels_3d"][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict["fp_" + box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts * 2 : self.fut_ts * 3]
            num_valid_ts = sum(gt_fut_masks == 1)
            if num_valid_ts == self.fut_ts:
                metric_dict["gt_" + box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict["cnt_ade_" + box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][: self.fut_ts * 2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox["trajs_3d"][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox["boxes_3d"][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict["ADE_" + box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict["cnt_fde_" + box_name] += 1
                    metric_dict["FDE_" + box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict["hit_" + box_name] += 1
                    else:
                        metric_dict["MR_" + box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self, pred_ego_fut_trajs, gt_ego_fut_trajs, gt_agent_boxes, gt_agent_feats, fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            "plan_L2_1s": 0,
            "plan_L2_2s": 0,
            "plan_L2_3s": 0,
            "plan_obj_col_1s": 0,
            "plan_obj_col_2s": 0,
            "plan_obj_col_3s": 0,
            "plan_obj_box_col_1s": 0,
            "plan_obj_box_col_2s": 0,
            "plan_obj_box_col_3s": 0,
        }
        metric_dict["fut_valid_flag"] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, "only support bs=1"
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i + 1) * 2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time],
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(), gt_ego_fut_trajs[:, :cur_time], occupancy
                )
                metric_dict["plan_L2_{}s".format(i + 1)] = traj_L2
                metric_dict["plan_obj_col_{}s".format(i + 1)] = obj_coll.mean().item()
                metric_dict["plan_obj_box_col_{}s".format(i + 1)] = obj_box_coll.mean().item()
            else:
                metric_dict["plan_L2_{}s".format(i + 1)] = 0.0
                metric_dict["plan_obj_col_{}s".format(i + 1)] = 0.0
                metric_dict["plan_obj_box_col_{}s".format(i + 1)] = 0.0

        return metric_dict

    def set_epoch(self, epoch):
        self.pts_bbox_head.epoch = epoch

    ### Modified train step to return all separate losses to compute grad norms ###
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        total_loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=total_loss,
            log_vars=log_vars,
            num_samples=len(data["img_metas"]),
            all_losses=losses,  # return all losses for grad norm computation
        )

        return outputs

        """
        Debug info of forward_train:
        pts_feats[0]: torch.Size([2, 6, 256, 12, 20]), len = 1
        gt_bboxes_3d[0]: LiDARInstance3DBoxes
        len(gt_labels_3d) = len(gt_bboxes_3d)
        map_gt_bboxes_3d[0], map_gt_labels_3d: mmdet3d_plugin.datasets.nuscenes_vad_dataset.LiDARInstanceLines
        img_metas[0].keys(): dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 'pts_filename', 'scene_token', 'can_bus', 'prev_bev'])
        
        (Pdb) p img_metas[0]
        {'filename': ['./data/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402938162460.jpg', './data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402938170339.jpg', './data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402938154844.jpg', './data/nuscenes/samples/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402938187525.jpg', './data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402938197423.jpg', './data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402938177893.jpg'], 
        'ori_shape': [(360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3)],
        'img_shape': [(384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3)], 
        'lidar2img': [array([[ 5.07915011e+02,  3.24258894e+02,  9.75765031e+00,
            -1.92790618e+02],
        [ 4.30379599e+00,  2.06350381e+02, -5.02656617e+02,
            -2.83051868e+02],
        [ 4.17915435e-03,  9.99804087e-01,  1.93474299e-02,
            -6.00969317e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]]), array([[ 5.46280894e+02, -2.45451992e+02, -1.17617172e+01,
            -1.59204503e+02],
        [ 1.60695496e+02,  1.20873815e+02, -5.03171218e+02,
            -3.05659496e+02],
        [ 8.37341787e-01,  5.46657159e-01,  4.96812843e-03,
            -6.87346981e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]])
        'pad_shape': [(384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3)], 
        'scale_factor': 1.0, 'box_mode_3d': <Box3DMode.LIDAR: 0>, 'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 
        'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}, 
        'sample_idx': 'fedfb3a6cb804635a0f47143f9ca8d6f', 'prev_idx': 'c844bf5a9f2243ff8f4bf2c85fe218ff', 'next_idx': '965f6af5a92449348409029a5f048a38', 
        'pts_filename': './data/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402938197897.pcd.bin', 'scene_token': 'cc8c0bf57f984915a77078b10eb33198', '
        can_bus': array([ 1.14795434e+00, -1.91956946e+00,  0.00000000e+00,  8.84410567e-01,
            8.84410567e-01,  8.84410567e-01,  8.84410567e-01, -1.78271547e-01,
            3.69926483e-01,  9.99809707e+00,  4.41768626e-03,  2.85090804e-02,
            2.18161628e-01,  4.37170480e+00,  0.00000000e+00,  0.00000000e+00,
            5.31327232e+00,  6.49850842e+00]), 'prev_bev': True}

        prev_bev.shape: torch.Size([2, 10000, 256])
        ego_his_trajs.shape: torch.Size([2, 1, 2, 2]
        ego_fut_trajs.shape: torch.Size([2, 1, 6, 2]))
        ego_fut_masks.shape: torch.Size([2, 1, 1, 6])
        (Pdb) p ego_fut_masks
        tensor([[[[1., 1., 1., 1., 1., 1.]]],
        [[[1., 1., 1., 1., 1., 1.]]]], device='cuda:0')
        (Pdb) p ego_fut_cmd
        tensor([[[[0., 1., 0.]]],
        [[[0., 0., 1.]]]], device='cuda:0')
        (Pdb) p ego_fut_cmd.shape
        torch.Size([2, 1, 1, 3])
        ego_lcf_feat.shape: torch.Size([2, 1, 1, 9])
        gt_attr_labels[0].shape: torch.Size([43, 34])



        p outs.keys()
        dict_keys(['bev_embed', 'all_cls_scores', 'all_bbox_preds', 'all_traj_preds', 'all_traj_cls_scores', 'map_all_cls_scores', 'map_all_bbox_preds', 'map_all_pts_preds', 'enc_cls_scores', 'enc_bbox_preds', 'map_enc_cls_scores', 'map_enc_bbox_preds', 'map_enc_pts_preds', 'ego_fut_preds'])
        outs['bev_embed'].shape: torch.Size([10000, 2, 256])
        
        p losses.keys()
        dict_keys(['loss_cls', 'loss_bbox', 'loss_traj', 'loss_traj_cls', 'loss_map_cls', 'loss_map_bbox', 'loss_map_iou', 'loss_map_pts', 'loss_map_dir', 'loss_plan_reg', 'loss_plan_bound', 'loss_plan_col', 'loss_plan_dir', 'd0.loss_cls', 'd0.loss_bbox', 'd1.loss_cls', 'd1.loss_bbox', 'd0.loss_map_cls', 'd0.loss_map_bbox', 'd0.loss_map_iou', 'd0.loss_map_pts', 'd0.loss_map_dir', 'd1.loss_map_cls', 'd1.loss_map_bbox', 'd1.loss_map_iou', 'd1.loss_map_pts', 'd1.loss_map_dir'])

        bev_gt_feat: torch.Size([101, 256])

        p agent_batch_idx
        tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        (Pdb) p agent_batch_idx.shape
        torch.Size([65])

        p lane_batch_idx
        tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        (Pdb) p lane_batch_idx.shape
        torch.Size([34])

        p bev_bev_feat[0].shape, len=2
        torch.Size([62, 256])
        """
