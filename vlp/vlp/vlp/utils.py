# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.nn.functional as F
from torch import distributed as dist
import torch
from torch import Tensor
from typing import Dict, List, Tuple, Union
from vlp.core.lib.utils import get_dist_info
from torch.distributed.nn.functional import all_gather  #  torch.distributed.all_gather is not differentiable!
import numpy as np


# TODO: should I add @torch.jit.script()?
def aggregate_bev_feats(
    bev_gt_feat: Tensor, bev_bev_feat: List[Tensor], agent_batch_idx: Tensor, lane_batch_idx: Tensor
) -> Tuple[Tensor, Tensor]:
    """Aggregate BEV (both orignal and text-based) features from all GPUs and bring them to the same shape according to procedure outlined in the VLP paper.
       Handles variable-length sequences (needs padding)

    Args:
        bev_gt_feat (torch.tensor): GT feature, (B + BM++ + BL++), 256]
        bev_bev_feat (List[Tensor]): BEV feature B x [(1+M+L), 256]
        agent_batch_idx (torch.tensor): agent batch id ts[000111bbb]
        lane_batch_idx (torch.tensor): lane batch id ts[000111bbb]

    Returns:
        Tuple[Tensor, Tensor]: Tuple of aggregated BEV features and GT features (e.g., text-based).
    """
    # [(B + BM++ + BL++), 256], B x [(1+M+L), 256], ts[000111bbb], ts[000111bbb]
    B = len(bev_bev_feat)
    bev_bev_feat = torch.cat(bev_bev_feat, dim=0)  # [B(1+M+L), 256]

    all_batch_idx = torch.cat([torch.arange(B), agent_batch_idx, lane_batch_idx], dim=0)  # [01b000111bb0011bb]
    bev_gt_feat = torch.cat([bev_gt_feat[all_batch_idx == bi] for bi in range(B)], dim=0)  # [B(1+M+L), 256]

    BML, C = bev_bev_feat.shape
    dtype = bev_bev_feat.dtype
    device = bev_bev_feat.device

    _, world_size = get_dist_info()

    # BEV GATHER
    bml = torch.tensor([BML], device=device)
    bml_all = all_gather(bml)
    bml_all = torch.cat(bml_all, dim=0)  # (bml)[GB,]
    bml_max = bml_all.max()

    bml_select = torch.zeros([bml_max], dtype=torch.bool, device=device)
    bml_select[:bml] = True
    bml_select_all = all_gather(bml_select)
    bml_select_all = torch.cat(bml_select_all, dim=0)  # [Gbml_max]

    pad_num = bml_max - bml
    # bev_bev_gt_feat = torch.cat([each['bev_bev_gt_feat'] for each in outs_contra_ls], dim=0) # [B(1+M+L), 256]
    # bev_text_gt_feat = torch.cat([bev_gt_feat[bev_batch_idx==bi] for bi in range(B)], dim=0) # [B(1+M+L), 256]
    pad_tensor = torch.zeros([pad_num, C], dtype=dtype, device=device)
    bev_bev_gt_feat_pad = torch.cat([bev_bev_feat, pad_tensor], dim=0)  # [bml_max, 256]
    bev_text_gt_feat_pad = torch.cat([bev_gt_feat, pad_tensor], dim=0)  # [bml_max, 256]

    # GATHER PAD
    bev_bev_gt_feat_pad_all = all_gather(bev_bev_gt_feat_pad)
    bev_text_gt_feat_pad_all = all_gather(bev_text_gt_feat_pad)

    bev_bev_gt_feat_pad_all = torch.cat(bev_bev_gt_feat_pad_all, dim=0)  # [Gbml_max, 256]
    bev_text_gt_feat_pad_all = torch.cat(bev_text_gt_feat_pad_all, dim=0)  # [Gbml_max, 256]

    bev_bev_gt_feat_all = bev_bev_gt_feat_pad_all[bml_select_all]  # [GB(1+M+L), 256]
    bev_text_gt_feat_all = bev_text_gt_feat_pad_all[bml_select_all]  # [GB(1+M+L), 256]

    assert bev_bev_gt_feat_all.requires_grad is True and bev_text_gt_feat_all.requires_grad is True

    return bev_bev_gt_feat_all, bev_text_gt_feat_all


def gather_tensor(tensor: Tensor, dim: int = 0) -> Tensor:
    """Gather tensors from all processes in a distributed setting.
    Assumes that the input tensor is the same shape across all processes.
    Args:
        tensor (torch.Tensor): The tensor to gather.
        dim (int): The dimension along which to concatenate the gathered tensors.
    Returns:
        torch.Tensor: The concatenated tensor from all processes.
    """
    world_size = dist.get_world_size()

    if not dist.is_initialized() or world_size == 1:
        return tensor

    gathered_tensors = all_gather(tensor)
    gathered_tensors = torch.cat(gathered_tensors, dim=dim)

    return gathered_tensors


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`."""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def set_requires_grad(tensor: Tensor) -> Union[bool, ValueError]:
    """Check if the tensor requires gradient."""
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Expected torch.Tensor, but got {type(tensor)}")

    if tensor.requires_grad:
        return True
    elif tensor.is_leaf:
        tensor.requires_grad = True
        return True
    else:
        raise ValueError("Tensor does not require gradient and is not a leaf tensor.")


def split_tensor_by_sorted_batch_idx(tensor, batch_idx):
    """
    Splits a tensor into batches using sorted batch indices.

    Args:
        tensor: Tensor of shape (N, ...)
        batch_idx: sorted list or tensor of shape (N,)

    Returns:
        List[Tensor] — one tensor per unique batch index
    """
    if not isinstance(batch_idx, torch.Tensor):
        batch_idx = torch.tensor(batch_idx, device=tensor.device)

    _, counts = torch.unique_consecutive(batch_idx, return_counts=True)
    splits = torch.split(tensor, counts.tolist(), dim=0)
    return splits
