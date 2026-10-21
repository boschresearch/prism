# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

""""Modified LLM encoder for SLP part of VLP framework."""

import copy
import torch.nn as nn
from transformers import GPT2TokenizerFast, GPT2Model
from clip import tokenize
from vlp.vlp.clip_text_head import CLIPTextHead
import torch
from typing import Dict, List, Tuple, Optional

from mmcv.utils import build_from_cfg, ConfigDict

from vlp.core.registry import VLP_MODULES
from vlp.core.encoders.base import DataEncoder
from vlp.core.encoders.llm_encoder import Mlp_for_adpt

# ============================================================================
# UTILITY CLASS
# ============================================================================


@VLP_MODULES.register_module()
class PromptGenerator_SLP:
    """Handles prompt generation for SLP part of VLP framework."""

    def __init__(
        self,
        use_modalities: Dict[str, bool] = dict(fut_traj=True, past_traj=False, ego_cmd=True),
        fut_ts: int = 6,
        past_ts: int = 2,
        **kwargs,
    ):

        assert set(use_modalities.keys()) == {
            "fut_traj",
            "past_traj",
            "ego_cmd",
        }, "use_modalities must contain exactly 'fut_traj', 'past_traj', 'ego_cmd' keys"

        assert fut_ts > 0 and fut_ts <= 6, f"Incorrect value for fut_ts: {fut_ts}"
        assert past_ts > 0 and past_ts <= 2, f"Incorrect value for past_ts: {past_ts}"

        self.use_modalities = use_modalities
        self.fut_ts = fut_ts
        self.past_ts = past_ts

        # Define idx2label mappings (for reference only)
        self.ego_command_map = {0: "turning right", 1: "turning left", 2: "going straight"}
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
        self.lane_label_map = {0: "a lane divider", 1: "a pedestrian crossing", 2: "a road boundary"}

        # Prompt templates
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompt templates based on configuration"""
        use_past_traj, use_fut_traj, use_ego_cmd = (
            self.use_modalities["past_traj"],
            self.use_modalities["fut_traj"],
            self.use_modalities["ego_cmd"],
        )

        if use_past_traj and use_fut_traj and use_ego_cmd:
            self.ego_prompt = f"The self-driving car is driving in an urban area. It is currently {{}}. The past trajectory of the car for the last {self.past_ts} timestamps was {{}}. The future trajectory of the car for the next {self.fut_ts} timestamps will be {{}}."

        elif use_fut_traj and use_ego_cmd:
            self.ego_prompt = f"The self-driving car is driving in an urban area. It is currently {{}}. The future trajectory of the car for the next {self.fut_ts} timestamps will be {{}}."

        elif use_past_traj and use_ego_cmd:
            self.ego_prompt = f"The self-driving car is driving in an urban area. It is currently {{}}. The past trajectory of the car for the last {self.past_ts} timestamps was {{}}."

        elif use_ego_cmd:
            self.ego_prompt = "The self-driving car is driving in an urban area. It is currently {}."
        else:
            raise ValueError("Ego prompts require either future trajectory or ego command to be True")

    def generate_ego_prompts(
        self, ego_fut_cmd: torch.Tensor, ego_fut_trajs: torch.Tensor, ego_his_trajs: torch.Tensor = None
    ) -> List[str]:
        """Generate ego prompts based on template.

        Returns:
            List of formatted ego prompts. Length is B.
        """
        B = ego_fut_cmd.shape[0]
        use_past_traj, use_fut_traj, use_ego_cmd = (
            self.use_modalities["fut_traj"],
            self.use_modalities["past_traj"],
            self.use_modalities["ego_cmd"],
        )

        if use_past_traj and use_fut_traj and use_ego_cmd:

            assert (
                ego_his_trajs is not None and ego_fut_trajs is not None and ego_fut_cmd is not None
            ), "All ego_his_trajs, ego_fut_trajs and ego_fut_cmd must be provided."

            plan_traj = copy.deepcopy(ego_fut_trajs[:, 0, : self.fut_ts, :2])  # e.g. [B, 6, 2]
            his_traj = copy.deepcopy(ego_his_trajs[:, 0, : self.past_ts, :2])  # e.g. [B, 2, 2]

            plan_traj = plan_traj.detach().cpu().numpy().round(4)
            his_traj = his_traj.detach().cpu().numpy().round(4)

            # ego-cmd + past_tray + fut_traj
            return [
                self.ego_prompt.format(
                    self.ego_command_map[int(ego_fut_cmd[bi, 0, 0].nonzero(as_tuple=True)[0])],
                    [list(his_traj[bi][ti]) for ti in range(self.past_ts)],
                    [list(plan_traj[bi][ti]) for ti in range(self.fut_ts)],
                )
                for bi in range(B)
            ]

        elif use_fut_traj and use_ego_cmd:

            assert (
                ego_fut_trajs is not None and ego_fut_cmd is not None
            ), "Both ego_fut_trajs and ego_fut_cmd must be provided when using future trajectory and command"

            # fut_ts = ego_fut_trajs.shape[2]
            plan_traj = copy.deepcopy(ego_fut_trajs[:, 0, : self.fut_ts, :2])  # [B, 6, 2]
            plan_traj = plan_traj.detach().cpu().numpy().round(4)

            # ego-cmd + fut_traj
            return [
                self.ego_prompt.format(
                    self.ego_command_map[int(ego_fut_cmd[bi, 0, 0].nonzero(as_tuple=True)[0])],
                    [list(plan_traj[bi][ti]) for ti in range(self.fut_ts)],
                )
                for bi in range(B)
            ]

        elif use_past_traj and use_ego_cmd:

            assert (
                ego_his_trajs is not None and ego_fut_cmd is not None
            ), "Both ego_his_trajs and ego_fut_cmd must be provided when using past trajectory and command"
            # past_ts = ego_his_trajs.shape[2]
            his_traj = copy.deepcopy(ego_his_trajs[:, 0, : self.past_ts, :2])  # [B, 2, 2]
            his_traj = his_traj.detach().cpu().numpy().round(4)
            # past_traj + ego-cmd
            return [
                self.ego_prompt.format(
                    self.ego_command_map[int(ego_fut_cmd[bi, 0, 0].nonzero(as_tuple=True)[0])],
                    [list(his_traj[bi][ti]) for ti in range(self.past_ts)],
                )
                for bi in range(B)
            ]

        elif use_ego_cmd and ego_fut_cmd is not None:
            # command only
            return [
                self.ego_prompt.format(self.ego_command_map[int(ego_fut_cmd[bi, 0, 0].nonzero(as_tuple=True)[0])])
                for bi in range(B)
            ]
        else:
            raise ValueError("Ego prompts require either future trajectory or ego command to be True and not None")
            # return ["" for _ in range(B)]


# ============================================================================
# ENCODER IMPLEMENTATION
# ============================================================================


@VLP_MODULES.register_module()
class LLMEncoder_SLP(DataEncoder):
    """LLM-based text encoder. Adapted for SLP part of VLP framework."""

    def __init__(
        self,
        prompt_generator: ConfigDict,
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
            self.llm_egoagent_adpt = Mlp_for_adpt(self.llm_dim, 1024, embed_dims)
            self.llm_egolane_adpt = Mlp_for_adpt(
                self.llm_dim, 1024, embed_dims
            )  # TODO: originally, it was 2*embed_dims, but it is not clear why
        else:
            raise NotImplementedError("Feature adaptation is required for original SLP LLM encoder")
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
        ego_his_trajs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract text features from all inputs

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Ego agent features [B, embed_dims]
            - Ego lane features  [B, embed_dims]
        """
        if device is None and gt_labels_3d:
            device = gt_labels_3d[0].device
            self.llm_head.to(device)
            self.llm_egoagent_adpt.to(device)
            self.llm_egolane_adpt.to(device)

        # Generate prompts
        ego_sentences = self.prompt_generator.generate_ego_prompts(
            ego_fut_cmd, ego_fut_trajs, ego_his_trajs=ego_his_trajs
        )
        # Encode text
        text_features = self._encode_text(ego_sentences, device)
        # Adapt features to target dimension
        egoagent_feat = self.llm_egoagent_adpt(text_features)
        egolane_feat = self.llm_egolane_adpt(text_features)

        return egoagent_feat, egolane_feat

    def _encode_text(self, sentences: List[str], device: torch.device) -> torch.Tensor:
        """Encode text sentences to features"""
        if self.llm_model == "gpt2":
            with torch.no_grad():
                inputs = self.llm_tokenizer(sentences, padding=True, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                outputs = self.llm_head(**inputs)
                features = outputs.last_hidden_state.max(1)[0]  # [B, 768]
        else:
            tokens = tokenize(sentences, context_length=self.context_length).to(device)
            features = self.llm_head(tokens)

        return features
