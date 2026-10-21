# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

""" Conditional Unscented Autoencoder.
Copyright (c) 2024 Robert Bosch GmbH
@author: Faris Janjos
@author: Marcel Hallgarten
@author: Anthony Knittel
@author: Maxim Dolgov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Optional, Union, Tuple, List, Dict
import math

import torch
import torch.nn as nn


from torch import nn
from vlp.core.lib.utils import class_from_path
from vlp.core.lib.matrix_utils import build_tril_matrix
from torch import Tensor

from vlp.core.registry import VLP_MODULES


def _is_reluish(m: nn.Module) -> bool:
    return isinstance(m, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Mish))


def _is_norm(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d))


class FlexibleMLP(nn.Module):
    """
    Flexible multi-layer perceptron with tunable activation and normalization after every hidden layer.

    Starts with an input dim, adds a variable number of linear layers with activation and normalization, ends with an optional output dim:
    input_dim -> [lin1 -> act1 -> norm1 -> lin2 -> ... -> actN -> normN -> linN] { -> output_dim }
    """

    def __init__(
        self,
        input_dim: int,
        lin_layers: "List[Dict]",
        output_dim: Union[int, None] = None,
    ) -> None:
        super().__init__()
        self.layers = []
        if not lin_layers:
            assert output_dim is not None, "output_dim must be defined if no linear layers are provided"
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            dim = input_dim
            for layer in lin_layers:
                self.layers.append(nn.Linear(dim, layer["dim"], bias=False))
                if layer["norm"] is not None:
                    self.layers.append(class_from_path(layer["norm"])(layer["dim"]))
                self.layers.append(class_from_path(layer["activation"])())
                dim = layer["dim"]

            if output_dim is not None:
                self.layers.append(nn.Linear(dim, output_dim))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @torch.no_grad()
    def init_weights(self, final: str = "xavier", zero_bias: bool = False) -> None:
        """
        final: "xavier" or "zeros" for the last Linear's WEIGHT.
        zero_bias: if True -> bias=0; else -> PyTorch-like default U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        """
        mods = list(self.layers)
        linear_ids = [i for i, m in enumerate(mods) if isinstance(m, nn.Linear)]
        last_linear = linear_ids[-1] if linear_ids else None

        for i, m in enumerate(mods):
            if isinstance(m, nn.Linear):
                is_last = i == last_linear

                # ---- weights ----
                if is_last:
                    if final == "zeros":
                        nn.init.zeros_(m.weight)
                    else:  # "xavier"
                        nn.init.xavier_uniform_(m.weight, gain=1.0)
                else:
                    # relu-ish -> kaiming, else xavier
                    act = None
                    for j in range(i + 1, len(mods)):
                        if isinstance(mods[j], nn.Linear):
                            break
                        if _is_norm(mods[j]):
                            continue
                        act = mods[j]
                        break

                    if _is_reluish(act):
                        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                    else:
                        nn.init.xavier_uniform_(m.weight, gain=1.0)

                # ---- bias ----
                if m.bias is not None:
                    if zero_bias:
                        nn.init.zeros_(m.bias)
                    else:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                        nn.init.uniform_(m.bias, -bound, bound)

            elif _is_norm(m):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)


@VLP_MODULES.register_module()
class NormalDistributionMLP(nn.Module):
    """
    Simple policy to predict moments of a multivariate Gaussian distribution.
    The MLP structure is shared apart from the last layer predicting means / log variances / correlations.
    If zero_correlation == True, then all correlations are set to zero and a diagonal-covariance distribution is modeled. Otherwise, full covariance matrix is modeled.

    The outputs are the mean vector and a lower triangular tril matrix L, Sigma=LL^T.
    """

    def __init__(
        self,
        conditional_input_dim: int,
        mu_dim: int,
        tril_matrix_dim: Tuple[int, int],
        hidden_layer_dims: "list[dict[str, str]]",
        fix_identity_cov: Optional[bool] = False,
        zero_correlation: Optional[bool] = False,
    ) -> None:
        """
        Args:
            hidden_layer_dims: Tunes network linear layer structure, is a list of {"dim": linear layer dim, "norm": normalization layer, "activation": activation function}
            fix_identity_cov: If True, the covariance matrix is fixed to the identity matrix and is not learned.
            zero_correlation: Tunes diagonal or full covariance. Works only if fix_identity_cov is False.
        """
        super().__init__()

        self.conditional_input_dim = conditional_input_dim
        self.mu_dim = mu_dim
        self.tril_matrix_dim = tril_matrix_dim

        assert self.tril_matrix_dim[1] == self.tril_matrix_dim[0], "tril matrix must be quadratic"
        assert (
            self.tril_matrix_dim[0] == self.mu_dim
        ), "dimension of the normal distribution must be equal for mu and tril"

        self.learn_tril = not fix_identity_cov
        self.include_correlation = not zero_correlation

        self.latent_distribution_dim = self.mu_dim
        self.corr_dim = (self.latent_distribution_dim * (self.latent_distribution_dim - 1)) // 2

        self.layers = FlexibleMLP(input_dim=self.conditional_input_dim, lin_layers=hidden_layer_dims)

        self.mu_layer = nn.Linear(hidden_layer_dims[-1]["dim"], self.latent_distribution_dim)
        if self.learn_tril:
            self.logvar_layer = nn.Linear(hidden_layer_dims[-1]["dim"], self.latent_distribution_dim)
        if self.learn_tril and self.include_correlation:
            self.corr_layer = nn.Linear(hidden_layer_dims[-1]["dim"], self.corr_dim)

        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Dict[str, torch.Tensor]:

        layer_output = self.layers(x)
        mu = self.mu_layer(layer_output)

        if self.learn_tril:
            logvar = self.logvar_layer(layer_output)
        else:
            batch_size = layer_output.shape[0]
            logvar = torch.zeros(size=[batch_size, self.latent_distribution_dim], device=layer_output.device)

        if self.include_correlation:
            corr = self.tanh(self.corr_layer(layer_output))
        else:
            batch_size = layer_output.shape[0]
            corr = torch.zeros(size=[batch_size, self.corr_dim], device=layer_output.device)

        tril = build_tril_matrix(logvar=logvar, corr=corr)

        return {"mu": mu, "tril_matrix": tril}


@VLP_MODULES.register_module()
class GMMDistributionPredictor(nn.Module):
    """
    Simple policy to predict parameters of a GMM of multivariate Gaussian distributions.
    Consists of n_components NormalDistributionsMLPs and one MLP to predict the components weights.

    The returned parameters are the mean vectors, lower triangular tril matrices L, Sigma=LL^T and the components weights
    """

    def __init__(
        self,
        conditional_input_dim: int,
        mu_dim: int,
        tril_matrix_dim: List[int],
        hidden_layer_dims: List[Dict],
        fix_identity_cov: bool,
        zero_correlation: bool,
        n_components: int,
    ) -> None:
        super().__init__()

        self.component_normal_predictors = torch.nn.ModuleList(
            [
                NormalDistributionMLP(
                    conditional_input_dim=conditional_input_dim,
                    mu_dim=mu_dim,
                    tril_matrix_dim=tril_matrix_dim,
                    hidden_layer_dims=hidden_layer_dims,
                    fix_identity_cov=fix_identity_cov,
                    zero_correlation=zero_correlation,
                )
                for _ in range(n_components)
            ]
        )

        self.components_weights_predictor = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            FlexibleMLP(
                input_dim=(
                    conditional_input_dim + 2 * mu_dim
                ),  # weights predictor doesn't consider covariances, input: mean and std of each component [B, N, 2*D+C]
                lin_layers=hidden_layer_dims,
            ),
            nn.Unflatten(dim=0, unflattened_size=[-1, n_components]),
            nn.Linear(hidden_layer_dims[-1]["dim"], 1),  # output: weight for each component [B, N, 1]
            nn.Flatten(start_dim=-2, end_dim=-1),  # after flattening last dimension to [B,N] softmax is applied
            nn.Softmax(dim=-1),
        )

        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run module

        returns: mu: [B, N, D], tril: [B, N, D, D], components_weights: [B, N]
            N: n_components, D: dimension of latent distribution
        """
        # infer mixture components
        components_params = [cp.forward(x) for cp in self.component_normal_predictors]

        # stack loc and scale so that mixture component is rightmost dimension (i.e. dim=1)
        components_mu = torch.stack([cp["mu"] for cp in components_params], dim=1)
        components_tril = torch.stack([cp["tril_matrix"] for cp in components_params], dim=1)

        weigths_predictor_inputs = torch.cat(
            [
                components_mu,
                # components weights predictor doesn't consider correlations
                # only use diagonal of tril
                torch.diagonal(components_tril, dim1=-2, dim2=-1),
                torch.tile(x[:, None, :], (1, self.n_components, 1)),
            ],
            dim=-1,
        )
        components_weights = self.components_weights_predictor.forward(weigths_predictor_inputs)

        return {
            "mu": components_mu,
            "tril_matrix": components_tril,
            "component_weights": components_weights,
        }


# --- Test Script ---
# Create FlexibleMLP config
flexible_mlp_config = {
    "input_dim": 224,
    "lin_layers": [
        {"dim": 112, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"},
        {"dim": 64, "norm": None, "activation": "torch.nn.Tanh"},
    ],
    "output_dim": 64,
}

# Create NormalDistributionMLP config
normal_dist_mlp_config = {
    "conditional_input_dim": 224,
    "mu_dim": 32,
    "tril_matrix_dim": [32, 32],
    "hidden_layer_dims": [
        {"dim": 112, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"},
        {"dim": 64, "norm": None, "activation": "torch.nn.Tanh"},
    ],
    "fix_identity_cov": False,
    "zero_correlation": False,
}

# Create GMMDistributionPredictor config
gmm_config = {
    "conditional_input_dim": 224,
    "mu_dim": 32,
    "tril_matrix_dim": [32, 32],
    "hidden_layer_dims": [
        {"dim": 112, "norm": "torch.nn.BatchNorm1d", "activation": "torch.nn.ReLU"},
        {"dim": 64, "norm": None, "activation": "torch.nn.Tanh"},
    ],
    "fix_identity_cov": False,
    "zero_correlation": False,
    "n_components": 9,
}

# Instantiate models
flexible_mlp = FlexibleMLP(
    input_dim=flexible_mlp_config["input_dim"],
    lin_layers=flexible_mlp_config["lin_layers"],
    output_dim=flexible_mlp_config["output_dim"],
)

normal_dist_mlp = NormalDistributionMLP(
    conditional_input_dim=normal_dist_mlp_config["conditional_input_dim"],
    mu_dim=normal_dist_mlp_config["mu_dim"],
    tril_matrix_dim=normal_dist_mlp_config["tril_matrix_dim"],
    hidden_layer_dims=normal_dist_mlp_config["hidden_layer_dims"],
    zero_correlation=normal_dist_mlp_config["zero_correlation"],
)

gmm_predictor = GMMDistributionPredictor(
    conditional_input_dim=gmm_config["conditional_input_dim"],
    mu_dim=gmm_config["mu_dim"],
    tril_matrix_dim=gmm_config["tril_matrix_dim"],
    hidden_layer_dims=gmm_config["hidden_layer_dims"],
    fix_identity_cov=gmm_config["fix_identity_cov"],
    zero_correlation=gmm_config["zero_correlation"],
    n_components=gmm_config["n_components"],
)

if __name__ == "__main__":
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, flexible_mlp_config["input_dim"])
    print(f"Test of FlexibleMLP, NormalDistributionMLP, and GMMDistributionPredictor\n")
    print(f"Input tensor shape: {input_tensor.shape}")

    # Perform inference on FlexibleMLP
    flexible_mlp_output = flexible_mlp(input_tensor)
    print(f"FlexibleMLP final output shape: {flexible_mlp_output.shape}\n")

    # Perform inference on NormalDistributionMLP
    normal_dist_mlp_output = normal_dist_mlp(input_tensor)
    print(
        f"NormalDistributionMLP final output: mu shape - {normal_dist_mlp_output['mu'].shape}, tril_matrix shape - {normal_dist_mlp_output['tril_matrix'].shape}\n"
    )

    # Perform inference on GMMDistributionPredictor
    gmm_output = gmm_predictor(input_tensor)
    print(
        f"GMMDistributionPredictor final output: mu shape - {gmm_output['mu'].shape}, "
        f"tril_matrix shape - {gmm_output['tril_matrix'].shape}, "
        f"component_weights shape - {gmm_output['component_weights'].shape}"
    )
    print(gmm_predictor)
    # --- Test Script End ---
