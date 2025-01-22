from typing import Optional, Union

import numpy as np


def _kmeans_plus_plus(
    X: np.ndarray,
    n_clusters: Union[int, np.uint] = 2,
    random_state: Optional[np.random.RandomState] = None,
):
    """Initializes cluster centers {yₖ} using an empirical probability distribution based on point contributions to
    inertia for selecting initial centroids. The first center y₁ being uniformly sampled from {ξᵢ} and subsequent
    centers {y₂, …, yₖ} are sampled with probabilities:

    qⱼ = min₍₁≤ₛ≤ₖ₎ ‖ξⱼ - yₛ⁰‖² ∕ Σᵢ₌₁ᴵ min₍₁≤ₛ≤ₖ₎ ‖ξᵢ - yₛ⁰‖²

    For the number of initial cluster centers `k`, the convergence rate to a local optimum is estimated as
    𝔼[F] ≤ 8(ln k + 2)F*, where F* denotes an optimal solution.

    Parameters
    ----------
    X : np.ndarray
        The input tensor containing training element {ξᵢ}.

    n_clusters : int or np.uint, default=2
        The number of initial cluster centers {yₖ}. Must be greater than or equal to 1.

    random_state : np.random.RandomState, optional
        Random state for reproducibility. If seed is None, return the RandomState singleton used by `np.random`.

    Returns
    -------
    cluster_centers : np.ndarray
        Initial positions of cluster centers {yₖ}.

    """

    X_len, _ = X.shape

    if not X_len:
        raise ValueError("The input tensor X should not be empty.")

    if n_clusters < 1:
        raise ValueError(
            "The number of initial cluster centers should not be less than 1."
        )

    if random_state is None:
        random_state = np.random.RandomState()

    random_indices = random_state.choice(X_len, size=1, replace=False)
    cluster_centers = np.expand_dims(X[random_indices.item()], axis=0)

    for _ in range(1, n_clusters):
        pairwise_distance = np.min(
            np.linalg.norm(X[:, np.newaxis] - cluster_centers, axis=-1), axis=-1
        )
        pairwise_probabilities = pairwise_distance / np.sum(pairwise_distance)
        cumulative_probabilities = np.cumsum(pairwise_probabilities)

        next_centroid_index = np.searchsorted(
            cumulative_probabilities, random_state.rand()
        )
        next_centroid = X[next_centroid_index]

        cluster_centers = np.vstack((cluster_centers, next_centroid))

    return cluster_centers
