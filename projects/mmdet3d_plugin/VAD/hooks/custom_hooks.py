# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from VAD
#   (https://github.com/hustvl/VAD, version 36047b6)
# Copyright (c) VAD authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time
from mmcv.parallel import is_module_wrapper
from typing import Optional, Sequence
from mmcv.runner import Runner
from torch.nn.utils import clip_grad
import torch


@HOOKS.register_module()
class TransferWeight(Hook):
    def __init__(self, every_n_inters=1):
        self.every_n_inters = every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class CustomSetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)


# TODO: not finished
@HOOKS.register_module()
class MemoryProfilerHook(Hook):
    """Memory profiler hook recording memory information including virtual
    memory, swap memory, and the memory of the current process.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval: int = 50) -> None:
        try:
            from psutil import swap_memory, virtual_memory

            self._swap_memory = swap_memory
            self._virtual_memory = virtual_memory
        except ImportError:
            raise ImportError("psutil is not installed, please install it by: " "pip install psutil")

        try:
            from memory_profiler import memory_usage

            self._memory_usage = memory_usage
        except ImportError:
            raise ImportError("memory_profiler is not installed, please install it by: " "pip install memory_profiler")

        self.interval = interval

    def _record_memory_information(self, runner: Runner) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the training or evaluation
                process.
        """
        # in Byte
        virtual_memory = self._virtual_memory()
        swap_memory = self._swap_memory()
        # in MB
        process_memory = self._memory_usage()[0]
        factor = 1024 * 1024
        runner.logger.info(
            "Memory information "
            "available_memory: "
            f"{round(virtual_memory.available / factor)} MB, "
            "used_memory: "
            f"{round(virtual_memory.used / factor)} MB, "
            f"memory_utilization: {virtual_memory.percent} %, "
            "available_swap_memory: "
            f"{round((swap_memory.total - swap_memory.used) / factor)}"
            " MB, "
            f"used_swap_memory: {round(swap_memory.used / factor)} MB, "
            f"swap_memory_utilization: {swap_memory.percent} %, "
            "current_process_memory: "
            f"{round(process_memory)} MB"
        )

    def after_train_iter(
        self, runner: Runner, data_batch: Optional[dict] = None, outputs: Optional[dict] = None
    ) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if (runner.iter + 1) % self.interval == 0 if self.interval > 0 else False:
            self._record_memory_information(runner)

    def after_val_iter(self, runner: Runner, data_batch: Optional[dict] = None, outputs=None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[:obj:`DetDataSample`], optional):
                Outputs from model. Defaults to None.
        """
        if (runner.iter + 1) % self.interval == 0 if self.interval > 0 else False:
            self._record_memory_information(runner)

    def after_test_iter(self, runner: Runner, data_batch: Optional[dict] = None, outputs=None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[:obj:`DetDataSample`], optional):
                Outputs from model. Defaults to None.
        """
        if (runner.iter + 1) % self.interval == 0 if self.interval > 0 else False:
            self._record_memory_information(runner)


@HOOKS.register_module()
class OptimizerHook_Custom(Hook):
    def __init__(
        self,
        grad_clip: dict = None,
        log_grad_norms: dict = None,
    ) -> None:
        self.grad_clip = grad_clip
        self.log_grad_norms = log_grad_norms if log_grad_norms is not None else False

        if self.log_grad_norms:
            assert "loss_keys" in self.log_grad_norms, "log_grad_norms should contain loss_keys"
            assert "max_norm" in self.log_grad_norms, "log_grad_norms should contain max_norm"
            assert (
                self.log_grad_norms["loss_keys"] is not None
                and isinstance(self.log_grad_norms["loss_keys"], list)
                and len(self.log_grad_norms["loss_keys"]) > 0
            ), "loss_keys should be a non-empty list of strings"

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def grad_norm_for_loss(self, loss: torch.Tensor, params, norm_type: float = 2.0) -> float:
        """Compute grad norm for a single loss term without touching .grad buffers.

        Args:
            loss: a scalar tensor (one component of your loss dict).
            params: iterable of model parameters.
            norm_type: which p-norm to use (2.0 = L2, 1.0 = L1, float("inf") = max).

        Returns:
            float: global norm value.
        """
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,  # so you can still backward total loss later
            create_graph=False,
            allow_unused=True,
        )

        if norm_type == float("inf"):
            max_norm = max((g.detach().abs().max().item() for g in grads if g is not None), default=0.0)
            return max_norm

        total = 0.0
        for g in grads:
            if g is not None:
                param_norm = g.detach().norm(norm_type).item()
                total += param_norm ** norm_type
        return total ** (1.0 / norm_type)

    def after_train_iter(self, runner):
        outputs = runner.outputs

        # optionally log grad norm for every loss item - used for debugging, has multiple backward passes
        if self.log_grad_norms:
            params = [p for p in runner.model.parameters() if p.requires_grad]
            norm_type = self.grad_clip.get("norm_type", 2) if self.grad_clip else 2
            num_samples = outputs["num_samples"]

            for name, loss in outputs["all_losses"].items():
                if "loss" not in name:
                    continue
                assert torch.is_tensor(loss) and loss.requires_grad, "loss should be a tensor and require grad"
                gnorm = self.grad_norm_for_loss(loss, params, norm_type)
                runner.log_buffer.update({f"grad_norm/{name}": float(gnorm)}, num_samples)

        # main backward on total loss
        runner.optimizer.zero_grad()
        runner.outputs["loss"].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])

        runner.optimizer.step()
