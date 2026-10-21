# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import json
import time
import torch
from pathlib import Path
import os.path as osp
import os

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class GradientAnalysisHook(Hook):
    """
    MMCV Hook to analyze model gradients at a specific iteration.

    Args:
        target_iteration (int): The iteration number when to perform analysis (default: 15)
        save_subdir (str): Subdirectory name within experiment root to save results (default: "gradient_analysis")
    """

    def __init__(self, target_iteration=15, save_subdir="gradient_analysis", **kwargs):
        super(GradientAnalysisHook, self).__init__()

        # Set our custom parameters
        self.target_iteration = target_iteration
        self.save_subdir = save_subdir
        self.analysis_done = False

        # Handle standard MMCV hook parameters manually
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_experiment_root_dir(self, runner):
        """
        Dynamically fetch the experiment root directory from the runner's work_dir.

        Args:
            runner: MMCV runner instance

        Returns:
            str: Path to the experiment root directory
        """
        if hasattr(runner, "work_dir") and runner.work_dir:
            return runner.work_dir
        else:
            # Fallback to current directory if work_dir is not available
            return os.getcwd()

    def _analyze_model_gradients(self, runner, model, save_path):
        """
        Analyzes a PyTorch model's parameters and categorizes them based on gradient tracking status.

        Args:
            model: PyTorch model to analyze
            save_path: Path to save the JSON file

        Returns:
            Dictionary containing the categorized parameter names
        """

        # Initialize categories
        no_grad_tracking = []
        requires_grad_but_no_flow = []
        requires_grad_with_flow = []

        # Analyze each parameter
        for name, param in model.named_parameters():
            if not param.requires_grad:
                # Category 1: No gradient tracking
                no_grad_tracking.append(name)
            else:
                # Parameter requires grad, check if gradients exist and are non-zero
                if param.grad is None:
                    # Category 2: Requires grad but no gradients computed yet
                    requires_grad_but_no_flow.append(name)
                else:
                    # Check if gradient is all zeros
                    if torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-8):
                        # Category 2: Requires grad but gradient is zero
                        requires_grad_but_no_flow.append(name)
                    else:
                        # Category 3: Requires grad and has non-zero gradients
                        requires_grad_with_flow.append(name)

        # Create result dictionary
        result = {
            "iteration": self.target_iteration,
            "not_tracking_grad": no_grad_tracking,
            "requires_grad_but_no_flow_detected": requires_grad_but_no_flow,
            "requires_grad_with_flow_detected": requires_grad_with_flow,
            "summary": {
                "total_parameters": len(list(model.parameters())),
                "not_tracking_grad_count": len(no_grad_tracking),
                "requires_grad_but_no_flow_detected_count": len(requires_grad_but_no_flow),
                "requires_grad_with_flow_detected_count": len(requires_grad_with_flow),
            },
        }

        # Ensure directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)

        filename = f"gradient_analysis_iter_{self.target_iteration}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        full_save_path = osp.join(save_path, filename)

        # Save to JSON file
        with open(full_save_path, "w") as f:
            json.dump(result, f, indent=2)

        # Log the results
        runner.logger.info(f"Gradient Analysis saved to: {full_save_path}")
        runner.logger.info(f"Summary:")
        runner.logger.info(f"  Total parameters: {result['summary']['total_parameters']}")
        runner.logger.info(f"  No grad tracking: {result['summary']['not_tracking_grad_count']}")
        runner.logger.info(
            f"  Requires grad but no flow detected: {result['summary']['requires_grad_but_no_flow_detected_count']}"
        )
        runner.logger.info(
            f"  Requires grad with flow detected: {result['summary']['requires_grad_with_flow_detected_count']}"
        )

        return result

    def after_train_iter(self, runner):
        """
        Hook method called after each training iteration.

        Args:
            runner: MMCV runner instance
        """
        # Check if this is the target iteration and analysis hasn't been done yet
        if runner.iter == self.target_iteration and not self.analysis_done:
            # Get the experiment root directory dynamically
            exp_root_dir = self._get_experiment_root_dir(runner)

            # Create the full save path
            save_path = osp.join(exp_root_dir, self.save_subdir)

            runner.logger.info(f"Running gradient analysis at iteration {self.target_iteration}")
            runner.logger.info(f"Experiment root directory: {exp_root_dir}")
            runner.logger.info(f"Analysis will be saved to: {save_path}")

            # Perform the gradient analysis
            try:
                result = self._analyze_model_gradients(runner, runner.model, save_path)
                self.analysis_done = True
                runner.logger.info("Gradient analysis completed successfully!")

                # Optional: You can store the result in the runner for later access
                if not hasattr(runner, "gradient_analysis_results"):
                    runner.gradient_analysis_results = {}
                runner.gradient_analysis_results[self.target_iteration] = result

            except Exception as e:
                runner.logger.error(f"Error during gradient analysis: {str(e)}")
                raise e


# Usage example in your MMCV config file:
"""
# Add this to your config file's custom_hooks section:

custom_hooks = [
    dict(
        type='GradientAnalysisHook',
        target_iteration=15,  # Analyze at iteration 15
        save_subdir='gradient_analysis'  # Save in work_dir/gradient_analysis/
    ),
    # ... other hooks
]

# Or if you want to analyze at a different iteration:
custom_hooks = [
    dict(
        type='GradientAnalysisHook',
        target_iteration=100,  # Analyze at iteration 100
        save_subdir='debug_gradients'  # Save in work_dir/debug_gradients/
    ),
]
"""
