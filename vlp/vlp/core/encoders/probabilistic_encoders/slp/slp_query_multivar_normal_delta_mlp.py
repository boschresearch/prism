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
class SLPQueryMutivarNormalDeltaMLP(ProbabilisticDataEncoder):
    """
    Probabilistic Encoder for SLP Query using Multivariate Normal refinement.

    This encoder predicts delta corrections for a given query mean and scales
    the diagonal elements of the distribution's scale_tril. Optionally integrates
    extra modalities, such as ego commands, into the trajectory-based delta prediction.

    Notes:
        - Only the diagonal elements of scale_tril are predicted/updated.
        - The predicted variance scaling is ensured positive via Softplus.
    """

    def __init__(
        self,
        traj_dim: int = 2,
        traj_ts: int = 6,
        embed_dim: int = 256,
        mu_delta_lin_layers: List[Dict] = None,
        var_factor_lin_layers: List[Dict] = None,
        extra_modalities: Dict[str, bool] = {},
    ) -> None:
        """Initialize the encoder.

        Args:
            traj_dim: Dimensionality of trajectory points (default=2).
            traj_ts: Number of trajectory timesteps (default=6).
            embed_dim: Dimension of the output embedding (default=256).
            mu_delta_lin_layers: List of dictionaries specifying the MLP layers for delta mean prediction.
            var_factor_lin_layers: List of dictionaries specifying the MLP layers for variance scaling prediction.
            extra_modalities: Dictionary specifying which additional modalities to include (e.g., {'ego_cmd': True}).
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

        self.mu_delta = FlexibleMLP(
            input_dim=(self.input_dim * traj_ts),
            lin_layers=mu_delta_lin_layers,
            output_dim=embed_dim,
        )

        self.var_factor = FlexibleMLP(
            input_dim=(self.input_dim * traj_ts),
            lin_layers=var_factor_lin_layers,
            output_dim=embed_dim,
        )

        self.softplus = nn.Softplus()

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
            2. Merge the trajectory data with the one-hot encoded features.
            3. Encode the merged input to predict delta mean and variance scaling factor.
            4. Update the mean and scale_tril of the original distribution based on predictions.

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

        # Encode GT input
        mu_delta = self.mu_delta(traj_input)  # (B, embed_dims)
        var_factor = self.var_factor(traj_input)  # (B, embed_dims)
        var_factor = self.softplus(var_factor)  # Ensure positive factor

        # Update mean of the distribution
        assert query_source_dist.mean.shape == mu_delta.shape
        new_mu = query_source_dist.mean + mu_delta

        # Update diagonal components of tril
        scale_tril = query_source_dist.scale_tril
        tril_off_diag = torch.tril(scale_tril, diagonal=-1)
        scaled_diag = torch.diagonal(scale_tril, dim1=-2, dim2=-1) * var_factor
        new_scale_tril = tril_off_diag + torch.diag_embed(scaled_diag)

        dist = MultivariateNormal(loc=new_mu, scale_tril=new_scale_tril)
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
    """Run a simple test to verify the SLPQueryMutivarNormalDeltaMLP functionality."""
    # Config
    B, embed_dim, traj_ts = 4, 256, 6
    lin_layers = [{"dim": 512, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SLPQueryMutivarNormalDeltaMLP(
        embed_dim=embed_dim,
        mu_delta_lin_layers=lin_layers,
        var_factor_lin_layers=lin_layers,
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

    return refined_dist


if __name__ == "__main__":
    test()
