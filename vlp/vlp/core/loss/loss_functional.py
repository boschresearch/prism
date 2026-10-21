# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.nn.functional as F
from torch import distributed as dist
import torch
from torch import Tensor
from typing import Dict, List
from vlp.core.lib.utils import get_dist_info


# TODO: should I add @torch.jit.script()?
def contrastive_loss(
    source_feat: Tensor,
    target_feat: Tensor,
    logit_scale,
    loss_key: str = "alp_loss_contra",
    loss_key_suffix: str = "",
    loss_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Contrastive Loss. Original implementation from VLP paper.
        Important: the code assumes already aggregated features from all GPUs.
        ALSO SUPPORTS SLP.

    Args:
        source_feat (List[Tensor]): predicted features. For BEV case: [GB(1+M+L), 256]
        target_feat (torch.tensor): GT features. For BEV case: [GB(1+M+L), 256]
        logit_scale (torch.tensor): logit scale. For BEV case: [1]
        loss_key (str): Key for the loss in the returned dictionary. Default is "alp_loss_contra".
        loss_key_suffix (str): Suffix to append to the loss key. Default is an empty string.
        loss_weight (float): Weight for the loss. Default is 1.0.

    Returns:
        Dict[str, Tensor]: Dictionary containing the contrastive loss. Key defined by `loss_key`.
    """

    assert target_feat.requires_grad is True
    assert source_feat.requires_grad is True

    # l2 normalize
    source_feat, target_feat = map(lambda t: F.normalize(t, p=2, dim=-1), (source_feat, target_feat))

    # cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits = logit_scale * source_feat @ target_feat.t()  # [GB(1+M+L), GB(1+M+L)] # (bev text)
    labels = torch.arange(logits.shape[0], device=target_feat.device)  # [0,1,2...] # softlabels: img + text
    contra_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

    key = f"{loss_key}_{loss_key_suffix}" if loss_key_suffix else loss_key
    return {key: contra_loss * loss_weight}


def _kld_loss(
    posterior_mu: Tensor,
    posterior_cov: Tensor,
    prior_mu: Tensor,
    prior_cov: Tensor,
    return_posterior: bool = False,
    loss_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """
    Calculates the KL divergence between a prior distribution (GT) and a posterior distribution (prediction).
    Both distributions are multivariate Gaussians defined by their mean vectors and lower triangular covariance matrices.

    Args:
        posterior_mu (Tensor): Mean vector of the posterior distribution. Shape: [batch_size, D].
        posterior_cov (Tensor): Lower triangular covariance matrix of the posterior distribution. Shape: [batch_size, D, D]. L@L=Sigma²
        prior_mu (Tensor): Mean vector of the prior distribution. Shape: [batch_size, D].
        prior_cov (Tensor): Lower triangular covariance matrix of the prior distribution. Shape: [batch_size, D, D].
        loss_weight (float): Weight for the loss. Default is 1.0.

    Returns:
        Tensor: KL divergence between the posterior and prior distributions. Shape: [].
    """
    assert posterior_mu.ndim == 2, f"Posterior mean must be 2D. Got {posterior_mu.ndim}D."
    assert posterior_cov.ndim == 3, f"Posterior covariance must be 3D. Got {posterior_cov.ndim}D."
    assert prior_mu.ndim == 2, f"Prior mean must be 2D. Got {prior_mu.ndim}D."
    assert prior_cov.ndim == 3, f"Prior covariance must be 3D. Got {prior_cov.ndim}D."

    print(f"Requires grad posterior_mu: {posterior_mu.requires_grad}, posterior_cov: {posterior_cov.requires_grad}")
    print(f"Requires grad prior_mu: {prior_mu.requires_grad}, prior_cov: {prior_cov.requires_grad}")

    posterior = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=posterior_mu,
        scale_tril=posterior_cov,
    )
    prior = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=prior_mu,
        scale_tril=prior_cov,
    )

    loss = torch.distributions.kl.kl_divergence(p=posterior, q=prior)
    output = {"kld_loss": loss.mean() * loss_weight}

    if return_posterior:
        output["posterior"] = posterior

    return output


def kld_loss(
    posterior_mu: Tensor,
    posterior_cov: Tensor,
    prior_mu: Tensor,
    prior_cov: Tensor,
    return_posterior: bool = False,
    loss_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Calculates the KL divergence between a prior distribution (GT) and a posterior distribution (prediction).
    Both distributions are multivariate Gaussians defined by their mean vectors and lower triangular covariance matrices.
    Support distributed training.

    Args:
        posterior_mu (Tensor): Mean vector of the posterior distribution. Shape: [batch_size, D].
        posterior_cov (Tensor): Lower triangular covariance matrix of the posterior distribution. Shape: [batch_size, D, D]. L@L=Sigma²
        prior_mu (Tensor): Mean vector of the prior distribution. Shape: [batch_size, D].
        prior_cov (Tensor): Lower triangular covariance matrix of the prior distribution. Shape: [batch_size, D, D].

    Returns:
        Tensor: KL divergence between the posterior and prior distributions. Shape: [].
    """

    _, world_size = get_dist_info()

    if world_size > 1:
        # Gather the posterior and prior distributions from all processes
        posterior_mu_all = [torch.zeros_like(posterior_mu) for _ in range(world_size)]
        posterior_cov_all = [torch.zeros_like(posterior_cov) for _ in range(world_size)]
        prior_mu_all = [torch.zeros_like(prior_mu) for _ in range(world_size)]
        prior_cov_all = [torch.zeros_like(prior_cov) for _ in range(world_size)]

        dist.all_gather(posterior_mu_all, posterior_mu)
        dist.all_gather(posterior_cov_all, posterior_cov)
        dist.all_gather(prior_mu_all, prior_mu)
        dist.all_gather(prior_cov_all, prior_cov)

        # Concatenate the gathered tensors
        posterior_mu = torch.cat(posterior_mu_all, dim=0)
        posterior_cov = torch.cat(posterior_cov_all, dim=0)
        prior_mu = torch.cat(prior_mu_all, dim=0)
        prior_cov = torch.cat(prior_cov_all, dim=0)

    # Calculate the KL divergence
    return _kld_loss(
        posterior_mu, posterior_cov, prior_mu, prior_cov, return_posterior=return_posterior, loss_weight=loss_weight
    )


def mse_loss(
    input: Tensor, target: Tensor, loss_key: str = "mse_loss", loss_key_suffix: str = "", loss_weight: float = 1.0
) -> Dict[str, Tensor]:
    """Calculates the Mean Squared Error (MSE) loss between input and target tensors.

    Args:
        input (Tensor): Input tensor. Shape: [B, C].
        target (Tensor): Target tensor. Shape: [B, C].
        loss_key (str): Key for the loss in the returned dictionary. Default is "mse_loss".
        loss_key_suffix (str): Suffix to append to the loss key. Default is an empty string.
        loss_weight (float): Weight for the loss. Default is 1.0.

    Returns:
        Dict[str, Tensor]: Dictionary containing the MSE loss.
    """
    assert (
        input.shape == target.shape
    ), f"Input and target must have the same shape. Got {input.shape} and {target.shape}."

    # assert input.requires_grad is True - for some cases we dont need that
    assert target.requires_grad is True

    mse = F.mse_loss(input, target)

    key = f"{loss_key}_{loss_key_suffix}" if loss_key_suffix else loss_key
    return {key: mse * loss_weight}


#  --- Test --- #
if __name__ == "__main__":

    test_contrastive_loss = False
    test_kld_loss = False
    test_mse_loss = True

    if test_contrastive_loss:
        # Test contrastive loss
        batch_size = 4
        feature_dim = 256
        logit_scale = torch.tensor([1.0], requires_grad=True)

        source_feat = torch.randn(batch_size, feature_dim, requires_grad=True)
        target_feat = torch.randn(batch_size, feature_dim, requires_grad=True)

        loss_dict = contrastive_loss(source_feat, target_feat, logit_scale)
        print("Test Contrastive Loss")
        print("Source Features:", source_feat.shape)
        print("Target Features:", target_feat.shape)
        print("Contrastive Loss:", loss_dict["bev_contra_loss"])

    if test_kld_loss:
        # KL Divergence
        batch = 2
        dim = 3
        posterior_mu = torch.randn(batch, dim)

        # def identity matrix
        posterior_cov = torch.eye(dim).repeat(batch, 1, 1)  # [batch, dim, dim]
        prior_mu = torch.randn(batch, dim)
        prior_cov = torch.eye(dim).repeat(batch, 1, 1)  # [batch, dim, dim]
        # Ensure covariance matrices are positive definite
        posterior_cov = torch.matmul(posterior_cov, posterior_cov.transpose(1, 2))
        prior_cov = torch.matmul(prior_cov, prior_cov.transpose(1, 2))

        kld = kld_loss(posterior_mu, posterior_cov, prior_mu, prior_cov)
        print("Test KL Divergence")
        print("Posterior Mean:", posterior_mu)
        print("Posterior Covariance:", posterior_cov)
        print("Prior Mean:", prior_mu)
        print("Prior Covariance:", prior_cov)
        print("KL Divergence:", kld)

    if test_mse_loss:
        # Test MSE Loss
        input_tensor = torch.randn(4, 3)
        target_tensor = torch.randn(4, 3)

        loss = mse_loss(input_tensor, target_tensor)
        print("Test MSE loss")
        print(("Input Tensor:", input_tensor.shape))
        print("MSE Loss:", loss)

# --- Test --- #
