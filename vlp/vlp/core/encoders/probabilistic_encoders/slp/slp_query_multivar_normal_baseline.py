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
from torch.distributions import MultivariateNormal


@VLP_MODULES.register_module()
class SLPQueryMutivarNormalBaseline(ProbabilisticDataEncoder):
    """Baseline Probabilistic Encoder for SLP Query. Takes query (mu), predicts tril and initialize distribution.
    Does not any additional information. Note: predict diagonal elements of tril only.

    extra_modalities: Additional input modalities, e.g., {'ego_cmd': True}.
    """

    def __init__(
        self,
        embed_dim: int,
        logvar_lin_layers: List[Dict],
        mu_lin_layers: List[Dict] = None,
        extra_modalities: Dict[str, bool] = {},
        **kwargs,
    ) -> None:
        """
        Args:
            embed_dims: dimension of embedding space after
            logvar_lin_layers: List of linear layers to predict log variance.
        """
        super().__init__()

        self.extra_modalities = extra_modalities

        self.use_extra_modalities = True if True in extra_modalities.values() else False
        if self.use_extra_modalities:
            # Define command and agent label mappings
            self.cmd_map = {0: "turning right", 1: "turning left", 2: "going straight"}
            extra_input_dim = self._get_extra_input_dim()
        else:
            extra_input_dim = 0

        self.input_dim = embed_dim + extra_input_dim

        if mu_lin_layers is None:
            self.mu_layer = nn.Identity()
        elif isinstance(mu_lin_layers, int):
            # single linear layer
            self.mu_layer = FlexibleMLP(
                input_dim=self.input_dim,
                lin_layers=None,
                output_dim=embed_dim,
            )
        else:
            self.mu_layer = FlexibleMLP(
                input_dim=self.input_dim,
                lin_layers=mu_lin_layers,
                output_dim=embed_dim,
            )

        self.logvar_layer = FlexibleMLP(
            input_dim=self.input_dim,
            lin_layers=logvar_lin_layers,
            output_dim=embed_dim,
        )

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

    def get_distribution(
        self,
        query: torch.Tensor,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MultivariateNormal:
        """Get the distribution parameters"""

        input = query

        if self.use_extra_modalities:
            if self.extra_modalities.get("ego_cmd", False):
                assert ego_fut_cmd is not None, "Ego future command is required but not provided."
                ego_cmd_one_hot = self._get_ego_cmd_one_hot(ego_fut_cmd)  # [B, one_hot_dim]
                input = torch.cat([input, ego_cmd_one_hot], dim=1)  # [B, Q + one_hot_dim]

        mu = self.mu_layer(input)
        logvar = self.logvar_layer(input)

        std = torch.exp(0.5 * logvar)
        tril = torch.diag_embed(std)

        dist = MultivariateNormal(loc=mu, scale_tril=tril)

        return dist


def test():
    """Run a simple test to verify the SLPQueryMutivarNormalBaseline functionality."""
    embed_dim = 256
    logvar_lin_layers = [{"dim": 256, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"}]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SLPQueryMutivarNormalBaseline(
        embed_dim=embed_dim,
        logvar_lin_layers=logvar_lin_layers,
    ).to(device)

    # Dummy data
    B = 4
    query = torch.randn(B, embed_dim).to(device)

    # Get distribution
    dist = model.get_distribution(query)

    # Sample from distribution
    samples = dist.rsample()

    print("Query Shape:", query.shape)
    print("Distribution Mean Shape:", dist.mean.shape)
    print("Distribution Covariance Shape:", dist.covariance_matrix.shape)
    print("Sample Shape:", samples.shape)
    print("Sampled Grad:", samples.requires_grad)
    print(f"Model: {model}")


if __name__ == "__main__":
    test()
