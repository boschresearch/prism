# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

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
class SLPQueryMutivarNormalMLP(ProbabilisticDataEncoder):
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
        mu_lin_layers: List[Dict] = None,
        var_lin_layers: List[Dict] = None,
        gt_data_lin_layers: List[Dict] = None,
        proj_lin_layers: List[Dict] = None,
        extra_modalities: Dict[str, bool] = {},
    ) -> None:
        """Initialize the encoder.

        Args:
            traj_dim: Dimensionality of trajectory points (default=2).
            traj_ts: Number of trajectory timesteps (default=6).
            embed_dim: Dimension of the output embedding (default=256).
            mu_lin_layers: MLP layers for original mean preprocessing (optional).
            var_lin_layers: MLP layers for original variance preprocessing (optional).
            gt_data_lin_layers: MLP layers for ground-truth data encoding (required).
            proj_lin_layers: MLP layers for final projection (required).
            extra_modalities: Additional input modalities, e.g., {'ego_cmd': True}.
        """
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_ts = traj_ts
        self.extra_modalities = extra_modalities

        self.use_extra_modalities = True if True in extra_modalities.values() else False
        if self.use_extra_modalities:
            # Define command and agent label mappings
            self.cmd_map = {0: "turning right", 1: "turning left", 2: "going straight"}
            extra_input_dim = self._get_extra_input_dim()
        else:
            extra_input_dim = 0

        self.input_dim = traj_dim + extra_input_dim

        self.mu_layers = (
            FlexibleMLP(
                input_dim=embed_dim,
                lin_layers=mu_lin_layers,
                output_dim=embed_dim,
            )
            if mu_lin_layers is not None
            else None
        )

        self.logvar_layers = (
            FlexibleMLP(
                input_dim=embed_dim,
                lin_layers=var_lin_layers,
                output_dim=embed_dim,
            )
            if var_lin_layers is not None
            else None
        )

        self.gt_data_mu_layers = FlexibleMLP(
            input_dim=(self.input_dim * traj_ts),
            lin_layers=gt_data_lin_layers,
            output_dim=embed_dim,
        )

        self.gt_data_logvar_layers = FlexibleMLP(
            input_dim=(self.input_dim * traj_ts),
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
        query_source_dist: MultivariateNormal,
        ego_fut_trajs: torch.Tensor,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Refine a query distribution using trajectory-based delta predictions.

        Args:
            query_source_dist: Original MultivariateNormal distribution [B, embed_dim].
            ego_fut_trajs: Ground truth ego trajectory [B, 1, T, traj_dim].
            ego_fut_cmd: Optional ego commands [B, 1, 1, cmd_dim].
            **kwargs: Additional unused keyword arguments.

        Procedure:
            1. Extract the ego future trajectories and one-hot encode the ego command.
            2. Merge the trajectory data with the one-hot features and encode them to GT features.
            3. (Optiolally) Preprocess the original mean and logvar using dedicated MLPs.
            4. Concatenate the GT features with the original features.
            5. Project the concatenated features to obtain refined mean and logvar.

        Returns:
            A refined MultivariateNormal distribution with updated mean and scale_tril.
        """

        assert self.use_extra_modalities, "Extra modalities are not enabled in the configuration."

        device = ego_fut_trajs.device
        B = ego_fut_trajs.shape[0]

        # Prepare GT input: trajectories and one-hot ego command (optional)
        ego_fut_trajs = ego_fut_trajs[:, 0, :, :]  # (B, T, 2)
        ego_one_hot = self._get_ego_cmd_one_hot(ego_fut_cmd)
        traj_input = self._merge_traj_with_one_hot(ego_fut_trajs, ego_one_hot)  # (B, T, 2 + one_hot_dim)
        traj_input = traj_input.reshape(B, -1)  # (B, T * (2 + one_hot_dim))
        assert (
            traj_input.shape[1] == self.input_dim * self.traj_ts
        ), f"Expected input shape (B, T * (2 + one_hot_dim)), got {ego_fut_trajs.shape}."

        # Encode GT input for mean and logvar predictions
        gt_features_mu = self.gt_data_mu_layers(traj_input)  # (B, embed_dims)
        gt_features_logvar = self.gt_data_logvar_layers(traj_input)  # (B, embed_dims)

        # Original mean and logvar
        source_mu = query_source_dist.mean
        source_logvar = 2 * torch.log(torch.diagonal(query_source_dist.scale_tril, dim1=-2, dim2=-1))  # (B, embed_dims)

        # Optionally preprocess original mean and logvar
        if self.mu_layers is not None:
            source_mu = self.mu_layers(source_mu)  # (B, embed_dims)

        if self.logvar_layers is not None:
            source_logvar = self.logvar_layers(source_logvar)  # (B, embed_dims)

        # Concatenate and project features
        combined_mu = torch.cat([gt_features_mu, source_mu], dim=-1)  # (B, embed_dims * 2)
        combined_logvar = torch.cat([gt_features_logvar, source_logvar], dim=-1)  # (B, embed_dims * 2)

        mu = self.proj_layers_mu(combined_mu)  # (B, embed_dims)
        logvar = self.proj_layers_logvar(combined_logvar)  # (B, embed_dims)

        # Convert logvar
        std = torch.exp(0.5 * logvar)
        scale_tril = torch.diag_embed(std)

        # Initialize distribution
        assert query_source_dist.mean.shape == mu.shape
        assert query_source_dist.scale_tril.shape == scale_tril.shape

        dist = MultivariateNormal(loc=mu, scale_tril=scale_tril)
        return dist

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
    """Test the SLPQueryMutivarNormalMLP functionality."""
    # Config
    B, embed_dim, traj_ts = 4, 256, 6
    lin_layers = [{"dim": 512, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SLPQueryMutivarNormalMLP(
        embed_dim=embed_dim,
        mu_lin_layers=lin_layers,
        var_lin_layers=lin_layers,
        gt_data_lin_layers=lin_layers,
        proj_lin_layers=lin_layers,
        extra_modalities={"ego_cmd": True},
    ).to(device)

    # Create test data
    source_mean = torch.randn(B, embed_dim).to(device)
    source_cov = torch.eye(embed_dim).unsqueeze(0).expand(B, -1, -1).to(device) + 0.01 * torch.eye(embed_dim).to(device)
    query_source_dist = MultivariateNormal(source_mean, torch.linalg.cholesky(source_cov))

    ego_fut_trajs = torch.randn(B, 1, traj_ts, 2).to(device)
    ego_fut_cmd = torch.eye(3)[torch.randint(0, 3, (B,))].view(B, 1, 1, 3).to(device)

    # Test
    refined_dist = model.get_distribution(query_source_dist, ego_fut_trajs, ego_fut_cmd)
    samples = refined_dist.rsample()

    # Verify
    mean_changed = not torch.allclose(refined_dist.mean, query_source_dist.mean, atol=1e-6)
    print(f"Sample shape: {samples.shape}")
    print(f"Distribution refined: {mean_changed}")
    print("✓ Test passed!" if mean_changed else "✗ Warning: No change detected!")

    print(f"Model: {model}")
    return refined_dist


if __name__ == "__main__":
    test()
