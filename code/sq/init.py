from typing import Optional, Union

import numpy as np

from scipy import optimize, sparse
from sklearn import metrics


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

    Raises
    ------
    ValueError
        If the input tensor {ξᵢ} does not contain any elements.

    ValueError
        The number of initial cluster centers {yₖ} is less than 1.
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


def _milp(
    X: np.ndarray,
    n_clusters: Union[int, np.uint] = 2,
    weights: Optional[np.ndarray] = None,
    **kwargs,
):
    X_len, _ = X.shape

    if not X_len:
        raise ValueError("The input tensor X should not be empty.")

    if n_clusters < 1:
        raise ValueError(
            "The number of initial cluster centers should not be less than 1."
        )

    if weights is None:
        # Assign default weights distribution as a normalized unit vector
        weights = np.ones(X_len) / X_len
    elif weights.ndim != 1:
        raise ValueError("The vector of element weights must have one dimension.")
    elif weights.shape != (X_len,):
        raise ValueError(
            "The number of weights should correspond to the size of the input tensor."
        )

    distance_coefficients = metrics.pairwise_distances(X).flatten()
    selection_coefficients = np.zeros(X_len)
    objective_function = np.concatenate((distance_coefficients, selection_coefficients))
    constraint_matrix = sparse.lil_matrix(
        (2 * X_len + 1, X_len * (X_len + 1)), dtype=np.float64
    )

    for i in range(X_len):
        constraint_matrix[2 * X_len, X_len * X_len + i] = 1.0
        constraint_matrix[X_len + i, X_len * X_len + i] = -1.0

        for j in range(X_len):
            constraint_matrix[j, i * X_len + j] = 1.0
            constraint_matrix[X_len + i, i * X_len + j] = 1.0

    integrality = np.array(
        [*np.zeros_like(distance_coefficients), *np.ones_like(selection_coefficients)]
    )
    bounds = optimize.Bounds(
        lb=np.full_like(objective_function, 0),
        ub=np.array(
            [
                *np.full_like(distance_coefficients, np.inf),
                *np.full_like(selection_coefficients, 1),
            ]
        ),
    )
    constraints = optimize.LinearConstraint(
        constraint_matrix.tocsr(),
        lb=np.array([*weights, *np.full(X_len, -np.inf), n_clusters]),
        ub=np.array([*weights, *np.full(X_len, 0), n_clusters]),
    )

    result = optimize.milp(
        c=objective_function,
        bounds=bounds,
        constraints=constraints,
        integrality=integrality,
        options=kwargs,
    )

    if not result.success:
        raise ValueError("The optimal solution of MILP initialization was not found.")

    # The optimal integer variable values function as selection indicators for the set of centers (1 = inclusion,
    # 0 = exclusion). Each variable corresponds to a distinct index position in the input tensor X.
    optimal_selection = result.x[-X_len:]
    center_indexes = np.argwhere(optimal_selection >= 0.5).flatten().tolist()
    cluster_centers = X[center_indexes, :]

    return cluster_centers
