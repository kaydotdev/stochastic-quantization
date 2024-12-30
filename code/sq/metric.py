from typing import Tuple

import numpy as np


def _calculate_loss(xi: np.ndarray, y: np.ndarray) -> np.float64:
    """Calculates stochastic Wasserstein (or Kantorovich–Rubinstein) distance between distributions ξ and y:

    F(y) = Σᵢ₌₁ᴵ pᵢ min₁≤k≤K d(ξᵢ, yₖ)ʳ

    Parameters
    ----------
    xi : np.ndarray
        The original distribution ξ with shape (N, D, ...).
    y : np.ndarray
        The quantized distribution y with shape (M, D, ...).

    Returns
    -------
    np.float64
        The calculated stochastic Wasserstein distance between distributions ξ and y.

    Raises
    ------
    ValueError
        If one of the distributions ξ or y is empty.
    ValueError
        If there is a shape mismatch between individual elements in distribution ξ and y.

    Notes
    -----
    The function assumes uniform weights (pᵢ = 1) for all elements in the original distribution.
    The exponent r in the formula is implicitly set to 1 in this implementation.
    """

    if xi.size == 0 or y.size == 0:
        raise ValueError("One of the distributions `xi` or `y` is empty.")

    if xi.shape[1:] != y.shape[1:]:
        raise ValueError(
            "The dimensions of individual elements in distribution `xi` and `y` must match. Elements in "
            f"`xi` have shape {xi.shape[1:]}, but y elements have shape {y.shape[1:]}."
        )

    pairwise_distance = np.linalg.norm(xi[:, np.newaxis] - y, axis=-1)
    min_distance = np.min(pairwise_distance, axis=-1)

    return np.sum(min_distance)


def _find_nearest_element(
    y: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.uint]:
    """Searches for the nearest element in `y` to `target` based on Euclidean distance. This function computes the
    Euclidean distance between each element in `y` and the `target`, then returns the element from `y` that has the
    smallest distance to `target`, along with its index. The shape of an individual element of `y` must match the
    shape of `target`.

    Parameters
    ----------
    y : np.ndarray
        The input tensor containing multiple elements to search from with shape (N, D, ...).

    target : np.ndarray
        The target tensor with shape (D, ...).

    Returns
    -------
    Tuple[np.ndarray, np.uint]
        A tuple containing two elements:

        1. np.ndarray: The nearest element found in `y` to the `target` with shape (D, ...).
        2. np.uint: The index of the nearest element in `y`.

    Raises
    ------
    ValueError
        If either the `y` input tensor or the `target` tensor is empty.
    ValueError
        If there is a shape mismatch between individual elements in `y` and `target`.
    """

    if y.size == 0 or target.size == 0:
        raise ValueError("Either the `y` input tensor or the `target` tensor is empty.")

    if y.shape[1:] != target.shape:
        raise ValueError(
            "The dimensions of individual elements in `y` and `target` must match. Elements in `y` have "
            f"shape {y.shape[1:]}, but `target` tensor has shape {target.shape}."
        )

    distance = np.linalg.norm(target - y, axis=1)
    nearest_index = np.argmin(distance)

    return y[nearest_index, :], nearest_index
