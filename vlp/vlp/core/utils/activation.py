# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Utilities for configuring activation functions."""

from typing import Callable, Dict

from torch import nn

supported_activations: Dict[str, Callable[[], nn.Module]] = {
    "tanh": lambda: nn.Tanh(),
    "elu": lambda: nn.ELU(inplace=True),
    "gelu": lambda: nn.GELU(),
    "relu": lambda: nn.ReLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
}


def assert_valid_activation(activation: str) -> None:
    """Check if the activation is supported.

    Args:
        activation (str): The name of the activation to check.
    """
    if not isinstance(activation, str):
        raise ValueError("Provided activation must be a string, got type: {type(activation)}.")
    if activation not in supported_activations:
        raise ValueError(
            f"Activation {activation} not supported.Supported activations are: {supported_activations.keys()}"
        )


def get_activation(activation: str) -> nn.Module:
    """Helper function that resolves an activation nn.Module from a provided str.

    Args:
        activation: activation function string

    Returns:
        an instance of the activation module
    """
    assert_valid_activation(activation)
    return supported_activations[activation]()
