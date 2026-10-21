# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

""""The difference from slp_query_multivar_normal_mlp.py is that this module uses raw ego_agent query to predict the distribution,
instead of using the ego_agent query distribution (mu and var)."""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from vlp.core.registry import VLP_MODULES
from vlp.core.prediction_models.heads import FlexibleMLP
from vlp.core.encoders.base import ProbabilisticDataEncoder
from vlp.vlp.utils import to_tensor
import copy
from torch import nn
from torch.distributions import MultivariateNormal, Distribution


@VLP_MODULES.register_module()
class SLPQueryMutivarNormalMLPfromQuery(ProbabilisticDataEncoder):
    """
    Probabilistic Encoder for SLP Query using Multivariate Normal refinement.

    This encoder predicts refined mean and log-variance for a multivariate normal distribution
    using ground-truth trajectory data (and optionally extra modalities like ego commands).

    Notes:
        - Only the diagonal elements of scale_tril are predicted/updated.
        - Model predicts log-variance and converts it to standard deviation via exp(0.5 * logvar).
    """

    def __init__(
        self,
        traj_dim: int = 2,
        traj_ts: int = 6,
        embed_dim: int = 256,
        query2mu_layers: List[Dict] = None,
        query2logvar_layers: List[Dict] = None,
        gt_data_lin_layers: List[Dict] = None,
        proj_lin_layers: List[Dict] = None,
        extra_modalities: Dict[str, bool] = {},
        freeze_posterior2prior_grad: bool = False,
        encode_three_traj: bool = False,
    ) -> None:
        """Initialize the encoder.

        Args:
            traj_dim: Dimensionality of trajectory points (default=2).
            traj_ts: Number of trajectory timesteps (default=6).
            embed_dim: Dimension of the output embedding (default=256).
            query2mu_layers: MLP layers for original mean preprocessing (optional).
            query2logvar_layers: MLP layers for original variance preprocessing (optional).
            gt_data_lin_layers: MLP layers for ground-truth data encoding (required).
            proj_lin_layers: MLP layers for final projection (required).
            extra_modalities: Additional input modalities, e.g., {'ego_cmd': True}.
            encode_three_traj: Whether to encode three trajectories based on the command (right, left, straight).
                               Used because VAD model is also trained to predict three trajectories and hardcode the selection by the command.
        """
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_ts = traj_ts
        self.extra_modalities = extra_modalities
        self.freeze_posterior2prior_grad = freeze_posterior2prior_grad
        self.encode_three_traj = encode_three_traj

        self.use_extra_modalities = True if True in extra_modalities.values() else False
        if self.use_extra_modalities:
            # Define command and agent label mappings
            self.cmd_map = {0: "turning right", 1: "turning left", 2: "going straight"}
            extra_input_dim = self._get_extra_input_dim()
        else:
            extra_input_dim = 0

        self.input_dim = traj_dim + extra_input_dim
        self.num_traj = 3 if self.encode_three_traj else 1

        self.query2mu_layers = (
            FlexibleMLP(
                input_dim=embed_dim,
                lin_layers=query2mu_layers,
                output_dim=embed_dim,
            )
            if query2mu_layers is not None
            else None
        )

        self.query2logvar_layers = (
            FlexibleMLP(
                input_dim=embed_dim,
                lin_layers=query2logvar_layers,
                output_dim=embed_dim,
            )
            if query2logvar_layers is not None
            else None
        )

        self.gt_data2mu_layers = FlexibleMLP(
            input_dim=(self.input_dim * traj_ts * self.num_traj),
            lin_layers=gt_data_lin_layers,
            output_dim=embed_dim,
        )

        self.gt_data2logvar_layers = FlexibleMLP(
            input_dim=(self.input_dim * traj_ts * self.num_traj),
            lin_layers=gt_data_lin_layers,
            output_dim=embed_dim,
        )

        self.proj_layers_mu = FlexibleMLP(
            input_dim=(embed_dim * 2),
            lin_layers=proj_lin_layers,
            output_dim=embed_dim,
        )

        self.proj_layers_logvar = FlexibleMLP(
            input_dim=(embed_dim * 2),
            lin_layers=proj_lin_layers,
            output_dim=embed_dim,
        )

    def get_distribution(
        self,
        ego_agent_query: torch.Tensor,
        ego_fut_trajs: torch.Tensor,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a Gaussian distribution over the ego-agent query embedding conditioned on
        ground-truth future trajectories (and optional ego command).

        This function encodes GT trajectory (and command) features and fuses them with
        the source query features to produce refined mean and (diagonal) covariance
        parameters of a MultivariateNormal.

        Args:
            ego_agent_query (Tensor): Source query features to refine, shape [B, D].
            ego_fut_trajs (Tensor): Ground-truth ego trajectory, shape [B, 1, T, traj_dim]
                (typically traj_dim=2). Only the first ego trajectory ([:, 0, ...]) is used.
            ego_fut_cmd (Tensor, optional): Ego command tensor, shape [B, 1, 1, C].
                It is converted to a one-hot representation via `_get_ego_cmd_one_hot(...)`.
            **kwargs: Unused.

        Returns:
            torch.distributions.MultivariateNormal:
                Batch of diagonal-covariance Gaussians with:
                - loc (mu):      [B, D]
                - scale_tril:    [B, D, D] (constructed as diag(exp(0.5 * logvar)))

        Notes:
            - Internally, the trajectory and (optional) command are merged and encoded
            into features for both mean and log-variance heads, concatenated with
            the corresponding query features, then projected to `mu` and `logvar`.
            - The covariance is diagonal (no inter-dimension correlations modeled).
        """

        use_fut_cmd = self.use_extra_modalities and self.extra_modalities.get("ego_cmd", False)

        device = ego_fut_trajs.device
        B = ego_fut_trajs.shape[0]

        ego_fut_trajs = ego_fut_trajs[:, 0, :, :]  # (B, T, 2)

        if self.encode_three_traj:
            ego_fut_trajs_all = self._extend_fut_traj(ego_fut_trajs, ego_fut_cmd[:, 0, 0])  # (B, T * 3, 2)
        else:
            ego_fut_trajs_all = ego_fut_trajs  # (B, T, 2)

        if use_fut_cmd:
            # Prepare GT input: one-hot ego command
            ego_one_hot = self._get_ego_cmd_one_hot(ego_fut_cmd)
            traj_input = self._merge_traj_with_one_hot(ego_fut_trajs_all, ego_one_hot)  # (B, T, 2 + one_hot_dim)
        else:
            traj_input = ego_fut_trajs_all

        traj_input = traj_input.reshape(B, -1)  # (B, T * (2 + 3)) or (B, 3 * T * 5)
        assert (
            traj_input.shape[1] == self.input_dim * self.traj_ts * self.num_traj
        ), f"Expected input shape (B, T * D * num_traj)), got {traj_input.shape}."

        # Encode GT input for mean and logvar features
        gt2mu_feat = self.gt_data2mu_layers(traj_input)  # (B, embed_dims)
        gt2logvar_feat = self.gt_data2logvar_layers(traj_input)  # (B, embed_dims)

        # Encode original query features for mean and logvar features
        if self.freeze_posterior2prior_grad:
            ego_agent_query = ego_agent_query.detach()

        source_mu_feat = self.query2mu_layers(ego_agent_query)  # (B, embed_dims)
        source_logvar_feat = self.query2logvar_layers(ego_agent_query)

        # Concatenate and project features
        combined_mu_feat = torch.cat([gt2mu_feat, source_mu_feat], dim=-1)  # (B, embed_dims * 2)
        combined_logvar_feat = torch.cat([gt2logvar_feat, source_logvar_feat], dim=-1)  # (B, embed_dims * 2)

        mu = self.proj_layers_mu(combined_mu_feat)  # (B, embed_dims)
        logvar = self.proj_layers_logvar(combined_logvar_feat)  # (B, embed_dims)

        # Convert logvar
        std = torch.exp(0.5 * logvar)
        scale_tril = torch.diag_embed(std)

        # Initialize distribution
        assert ego_agent_query.shape == mu.shape
        assert scale_tril.shape[1] == scale_tril.shape[2] == mu.shape[1]

        dist = MultivariateNormal(loc=mu, scale_tril=scale_tril)
        return dist

    def _extend_fut_traj(
        self,
        ego_fut_trajs: torch.Tensor,
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extend the future trajectory to three trajectories based on the command.
        Other trajectories, apart from the GT one, are zeros.

        Args:
            ego_fut_trajs: [B, T, 2]
            ego_fut_cmd:   [B, 3] one-hot (command)
        Returns:
            [B, 3*T, 2]
        """
        if ego_fut_trajs.ndim != 3:
            raise ValueError(f"ego_fut_trajs must be [B, T, 2], got {ego_fut_trajs.shape}")

        if ego_fut_cmd is None:
            raise ValueError("ego_fut_cmd must be provided (one-hot with 3 classes).")

        B, T, D = ego_fut_trajs.shape
        if D != 2:
            raise AssertionError("Trajectory dimension should be 2 (x,y).")

        if ego_fut_cmd.ndim == 2:
            ego_cmd_one_hot = ego_fut_cmd  # [B, 3]
        else:
            raise ValueError(f"ego_fut_cmd must be [B,3], got {ego_fut_cmd.shape}")

        # Argmax over commands -> index in {0,1,2}
        cmd_indices = torch.argmax(ego_cmd_one_hot, dim=1)  # [B]

        # Allocate output [B, 3, T, 2]
        out = torch.zeros((B, 3, T, D), device=ego_fut_trajs.device, dtype=ego_fut_trajs.dtype)

        batch_indices = torch.arange(B, device=ego_fut_trajs.device)
        out[batch_indices, cmd_indices] = ego_fut_trajs  # place GT traj into the chosen head

        # # Sanity check
        # for b in range(B):
        #     gt_traj = ego_fut_trajs[b]
        #     inserted_traj = out[b, cmd_indices[b]]
        #     if not torch.allclose(gt_traj, inserted_traj, atol=1e-6):
        #         raise RuntimeError(f"Sanity check failed: trajectory not correctly inserted for batch {b}.")
        #     # Ensure other trajectories are all zeros
        #     mask = torch.ones(3, dtype=torch.bool, device=ego_fut_trajs.device)
        #     mask[cmd_indices[b]] = False
        #     if torch.any(out[b, mask].abs() > 1e-8):
        #         raise RuntimeError(f"Sanity check failed: non-zero values found in other trajectories for batch {b}.")

        return out.reshape(B, 3 * T, D)

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
    """Test SLPQueryMutivarNormalMLPfromQuery.get_distribution with GT traj + ego cmd (raw query in)."""
    # Config
    torch.manual_seed(0)
    B, embed_dim, traj_ts, cmd_dim = 4, 256, 6, 3
    lin_layers = [{"dim": 512, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model (note: use query2* args, not mu_lin_layers/var_lin_layers)
    model = (
        SLPQueryMutivarNormalMLPfromQuery(
            traj_dim=2,
            traj_ts=traj_ts,
            embed_dim=embed_dim,
            query2mu_layers=lin_layers,
            query2logvar_layers=lin_layers,
            gt_data_lin_layers=lin_layers,
            proj_lin_layers=lin_layers,
            extra_modalities={"ego_cmd": True},
            encode_three_traj=True,
        )
        .to(device)
        .eval()
    )

    # Synthetic inputs
    ego_agent_query = torch.randn(B, embed_dim, device=device)  # [B, D]
    ego_fut_trajs = torch.randn(B, 1, traj_ts, 2, device=device)  # [B, 1, T, 2]
    # one-hot ego cmd: [B, 1, 1, C]
    idx = torch.randint(0, cmd_dim, (B,), device=device)
    ego_fut_cmd = torch.eye(cmd_dim, device=device)[idx].view(B, 1, 1, cmd_dim)

    # Call the method under test
    refined_dist = model.get_distribution(
        ego_agent_query=ego_agent_query,
        ego_fut_trajs=ego_fut_trajs,
        ego_fut_cmd=ego_fut_cmd,
    )

    # rsample to check gradient-friendly sampling works
    samples = refined_dist.rsample()  # [B, D]

    # Verifications
    assert refined_dist.loc.shape == (
        B,
        embed_dim,
    ), f"Expected loc shape {(B, embed_dim)}, got {refined_dist.loc.shape}"
    assert refined_dist.scale_tril.shape == (
        B,
        embed_dim,
        embed_dim,
    ), f"Expected scale_tril shape {(B, embed_dim, embed_dim)}, got {refined_dist.scale_tril.shape}"
    assert samples.shape == (B, embed_dim), f"Expected sample shape {(B, embed_dim)}, got {samples.shape}"

    # Diagonal std > 0 (since scale_tril is diag_embed(std))
    diag_std = torch.diagonal(refined_dist.scale_tril, dim1=-2, dim2=-1)
    assert torch.all(diag_std > 0), "All stds should be positive."

    # Basic sanity: distribution depends on inputs (not a constant mapping)
    with torch.no_grad():
        refined_dist_2 = model.get_distribution(
            ego_agent_query=torch.randn(B, embed_dim, device=device),
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
        )
        changed = not torch.allclose(refined_dist.loc, refined_dist_2.loc, atol=1e-6)

    print(f"Sample shape: {samples.shape}")
    print("Loc/scale_tril shapes OK and stds positive.")
    print(f"Distribution responds to input changes: {changed}")
    print("✓ Test passed!")

    return refined_dist


if __name__ == "__main__":
    test()
