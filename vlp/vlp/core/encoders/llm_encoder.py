# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import copy
import torch.nn as nn
from transformers import GPT2TokenizerFast, GPT2Model
from clip import tokenize
from vlp.vlp.clip_text_head import CLIPTextHead
import torch
from typing import Dict, List, Tuple, Optional

from mmcv.utils import build_from_cfg

from vlp.core.registry import VLP_MODULES
from vlp.core.encoders.base import DataEncoder

# ============================================================================
# UTILITY CLASSES
# ============================================================================


class Mlp_for_adpt(nn.Module):
    """Adapter for BEV"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Generic adapter for feature dimension adaptation

        Args:
            in_features (int): Input features dimension
            hidden_features (int, optional): Hidden feature dimension. Defaults to None.
            out_features (int, optional): Output dimension. Defaults to None.
            act_layer (_type_, optional): Activation layer. Defaults to nn.GELU.
            drop (int, optional): Dropout. Defaults to 0..
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@VLP_MODULES.register_module()
class PromptGenerator:
    """Handles prompt generation for different scenarios"""

    def __init__(self, future_traj: bool = False, map_info: bool = False):
        self.future_traj = future_traj
        self.map_info = map_info

        # Command mapping
        self.command_map = {0: "turning right", 1: "turning left", 2: "going straight"}

        # Agent label mapping (nuScenes 10 classes)
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

        # Lane label mapping
        self.lane_label_map = {0: "a lane divider", 1: "a pedestrian crossing", 2: "a road boundary"}

        # Prompt templates
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompt templates based on configuration"""
        if self.future_traj:
            # cmd + future trajectory
            self.ego_prompt = "The self-driving car is driving in an urban area. It is currently {}. The future trajectory of the car for the next 6 timestamps will be {}."
            # label + 10d bbox + future trajectory
            self.agent_prompt = "This is an agent around the self-driving car. Its label is {}, and its 3D bounding box is {}. The future trajectory of the agent for the next 6 timestamps will be {}."
        else:
            # cmd only
            self.ego_prompt = "The self-driving car is driving in an urban area. It is currently {}."
            # label + 10d bbox only
            self.agent_prompt = (
                "This is an agent around the self-driving car. Its label is {}, and its 3D bounding box is {}."
            )

        if self.map_info:
            # label + 4d bbox
            self.lane_prompt = "This is a lane instance around the self-driving car. It is {}. Its 2D bounding box on the BEV map is {}."
        else:
            self.lane_prompt = None

    def generate_ego_prompts(
        self, ego_fut_cmd: torch.Tensor, ego_fut_trajs: Optional[torch.Tensor] = None
    ) -> List[str]:
        """Generate ego prompts for batch"""
        B = ego_fut_cmd.shape[0]

        if self.future_traj and ego_fut_trajs is not None and ego_fut_cmd is not None:
            fut_ts = ego_fut_trajs.shape[2]
            plan_traj = copy.deepcopy(ego_fut_trajs[:, 0, :fut_ts, :2])  # [B, 6, 2]
            plan_traj = plan_traj.detach().cpu().numpy().round(4)

            # command + future trajectory
            return [
                self.ego_prompt.format(
                    self.command_map[int(ego_fut_cmd[bi, 0, 0].nonzero(as_tuple=True)[0])],
                    [list(plan_traj[bi][ti]) for ti in range(fut_ts)],
                )
                for bi in range(B)
            ]
        elif ego_fut_cmd is not None:
            # command only
            return [
                self.ego_prompt.format(self.command_map[int(ego_fut_cmd[bi, 0, 0].nonzero(as_tuple=True)[0])])
                for bi in range(B)
            ]
        else:
            return ["" for _ in range(B)]

    def generate_agent_prompts(
        self,
        gt_labels_3d: List[torch.Tensor],
        gt_bboxes_3d: List,
        fut_ts: int,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        pc_range: List[float] = None,
    ) -> Tuple[List[str], List[int]]:
        """Generate agent prompts for batch"""
        B = len(gt_labels_3d)
        agent_gt_labels = torch.cat(copy.deepcopy(gt_labels_3d), dim=0)

        # Create batch indices
        agent_batch_idx = []
        for bi in range(B):
            agent_batch_idx.extend([bi] * gt_labels_3d[bi].shape[0])

        BM = agent_gt_labels.shape[0]
        agent_box_9d = torch.cat([copy.deepcopy(gt_bboxes_3d[bi].tensor) for bi in range(B)], dim=0)
        agent_box_10d = self._normalize_to_10d_bbox(agent_box_9d, pc_range)
        agent_box_10d = agent_box_10d.detach().cpu().numpy().round(4)

        if self.future_traj:
            # label + 10d bbox + future trajectory
            agent_fut_traj = torch.cat([copy.deepcopy(gt_attr_labels[bi][:, : fut_ts * 2]) for bi in range(B)], dim=0)
            agent_fut_traj = agent_fut_traj.reshape(BM, fut_ts, 2).detach().cpu().numpy().round(4)

            agent_sentences = [
                self.agent_prompt.format(
                    self.agent_label_map[int(agent_gt_labels[ai])],
                    list(agent_box_10d[ai]),
                    [list(agent_fut_traj[ai, ti]) for ti in range(fut_ts)],
                )
                for ai in range(BM)
            ]
        else:
            agent_sentences = [
                self.agent_prompt.format(self.agent_label_map[int(agent_gt_labels[ai])], list(agent_box_10d[ai]))
                for ai in range(BM)
            ]

        return agent_sentences, agent_batch_idx

    def generate_lane_prompts(
        self,
        map_gt_labels_3d: Optional[List[torch.Tensor]],
        map_gt_bboxes_3d: Optional[List],
        pc_range: List[float] = None,
    ) -> Tuple[List[str], List[int]]:
        """Generate lane prompts for batch"""
        if not self.map_info or map_gt_labels_3d is None:
            return [], []

        B = len(map_gt_labels_3d)
        lane_gt_labels = torch.cat(map_gt_labels_3d, dim=0)

        # Create batch indices
        lane_batch_idx = []
        for bi in range(B):
            lane_batch_idx.extend([bi] * map_gt_labels_3d[bi].shape[0])

        lane_gt_bbox = torch.cat([copy.deepcopy(map_gt_bboxes_3d[bi].bbox) for bi in range(B)], dim=0)
        lane_gt_bbox = self._normalize_4d_lane_bbox(lane_gt_bbox, pc_range)
        BL = lane_gt_bbox.shape[0]
        lane_gt_bbox = lane_gt_bbox.detach().cpu().numpy().round(4)

        lane_sentences = [
            self.lane_prompt.format(self.lane_label_map[int(lane_gt_labels[ai])], list(lane_gt_bbox[ai]))
            for ai in range(BL)
        ]

        return lane_sentences, lane_batch_idx

    def _normalize_to_10d_bbox(self, bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
        """Normalize 9D bbox to 10D format"""
        cx = (bboxes[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        cy = (bboxes[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        cz = (bboxes[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
        w = bboxes[..., 3:4].log()
        l = bboxes[..., 4:5].log()
        h = bboxes[..., 5:6].log()
        rot = bboxes[..., 6:7]

        if bboxes.size(-1) > 7:
            vx = bboxes[..., 7:8]
            vy = bboxes[..., 8:9]
            normalized_bboxes = torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
        else:
            normalized_bboxes = torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1)

        return normalized_bboxes

    def _normalize_4d_lane_bbox(self, bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
        """Normalize 4D lane bbox"""
        bboxes[..., 0] = (bboxes[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        bboxes[..., 1] = (bboxes[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        bboxes[..., 2] = (bboxes[..., 2] - pc_range[0]) / (pc_range[3] - pc_range[0])
        bboxes[..., 3] = (bboxes[..., 3] - pc_range[1]) / (pc_range[4] - pc_range[1])
        return bboxes


# ============================================================================
# ENCODER IMPLEMENTATION
# ============================================================================


@VLP_MODULES.register_module()
class LLMEncoder(DataEncoder):
    """LLM-based text encoder"""

    def __init__(
        self,
        prompt_generator: Dict,
        llm_model: str = "clip",
        llm_model_pretrain: Optional[str] = "RN50x64.pt",
        embed_dims: int = 256,
        include_feat_adapter: bool = True,
    ):
        super().__init__(embed_dims)
        self.llm_model = llm_model

        # Initialize prompt generator
        self.prompt_generator = build_from_cfg(prompt_generator, VLP_MODULES)

        # Initialize LLM
        self._setup_llm(llm_model, llm_model_pretrain)

        # Feature adaptation layer
        if include_feat_adapter:
            self.llm_feat_adpt = Mlp_for_adpt(self.llm_dim, 1024, embed_dims)
        else:
            self.llm_feat_adpt = nn.Identity()

    def _setup_llm(self, llm_model: str, llm_model_pretrain: Optional[str]):
        """Setup LLM model and tokenizer"""
        if llm_model == "gpt2":
            print("\nUsing the default gpt2 LLM model: ")
            self.llm_head = GPT2Model.from_pretrained("gpt2")
            self.llm_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_dim = 768
        else:
            print("\nUsing CLIP model...")
            try:
                from clip import tokenize
            except ImportError:
                print("Try to pip install clip by: pip install git+https://github.com/openai/CLIP.git")

            llm_config = dict(
                context_length=250,
                vocab_size=49408,
                transformer_width=1024,
                transformer_heads=16,
                transformer_layers=12,
            )
            self.llm_head = CLIPTextHead(**llm_config)

            # Load pretrained weights
            self._load_pretrained_weights(llm_model_pretrain)

            self.llm_tokenizer = None
            self.llm_dim = self.llm_head.transformer_width
            self.context_length = self.llm_head.context_length

        # Freeze LLM parameters
        for param in self.llm_head.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, llm_model_pretrain: Optional[str]):
        """Load pretrained weights for CLIP model"""
        try:
            if llm_model_pretrain is not None:
                self.llm_head.load_state_dict(
                    torch.jit.load(llm_model_pretrain, map_location="cpu").state_dict(), strict=False
                )
        except Exception:
            try:
                import wget

                url = "https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/resolve/main/checkpoints/clip/RN50x64.pt"
                filename = wget.download(url)
                self.llm_head.load_state_dict(torch.jit.load(filename, map_location="cpu").state_dict(), strict=False)
            except Exception:
                print("Try to download the pretrained model and please install wget by: pip install wget")

    def get_features(
        self,
        device: torch.device = None,
        ego_fut_trajs: Optional[torch.Tensor] = None,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        gt_labels_3d: Optional[List[torch.Tensor]] = None,
        gt_bboxes_3d: Optional[List] = None,
        gt_attr_labels: Optional[List[torch.Tensor]] = None,
        map_gt_labels_3d: Optional[List[torch.Tensor]] = None,
        map_gt_bboxes_3d: Optional[List] = None,
        pc_range: List[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract text features from all inputs

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Combined features [(B + BM++ + BL++), embed_dims],
                                    (1 - ego, M - agents, L - lanes, ++ - varying size)
                - Agent batch indices
                - Lane batch indices
        """
        if device is None and gt_labels_3d:
            device = gt_labels_3d[0].device
            self.llm_head.to(device)
            self.llm_feat_adpt.to(device)
        fut_ts = ego_fut_trajs.shape[2] if ego_fut_trajs is not None else 0

        # Generate prompts
        ego_sentences = self.prompt_generator.generate_ego_prompts(ego_fut_cmd, ego_fut_trajs)
        agent_sentences, agent_batch_idx = self.prompt_generator.generate_agent_prompts(
            gt_labels_3d, gt_bboxes_3d, fut_ts, gt_attr_labels, pc_range
        )
        lane_sentences, lane_batch_idx = self.prompt_generator.generate_lane_prompts(
            map_gt_labels_3d, map_gt_bboxes_3d, pc_range
        )

        # Combine all sentences
        all_sentences = ego_sentences + agent_sentences + lane_sentences

        # Encode text
        text_features = self._encode_text(all_sentences, device)

        # Adapt features to target dimension
        adapted_features = self.llm_feat_adpt(text_features)

        return {
            "feats": adapted_features,
            "agent_batch_idx": torch.tensor(agent_batch_idx),
            "lane_batch_idx": torch.tensor(lane_batch_idx),
        }

    def _encode_text(self, sentences: List[str], device: torch.device) -> torch.Tensor:
        """Encode text sentences to features"""
        if self.llm_model == "gpt2":
            with torch.no_grad():
                inputs = self.llm_tokenizer(sentences, padding=True, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                outputs = self.llm_head(**inputs)
                features = outputs.last_hidden_state.max(1)[0]
        else:
            tokens = tokenize(sentences, context_length=self.context_length).to(device)
            features = self.llm_head(tokens)

        return features
