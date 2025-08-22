from typing import Optional, Union

import numpy as np

from scipy import optimize, sparse
from sklearn import metrics


def init_centroids(init: np.ndarray | str, n_clusters: int, x: np.ndarray, random_state: np.random.RandomState):
    x_len, x_dims = x.shape
    match init:
        case _ if isinstance(init, np.ndarray):
            init_len, init_dims = init.shape

            if init_dims != x_dims:
                raise ValueError(
                    f"The dimensions of initial quantized distribution ({init_len}) and input tensor "
                    f"({x_dims}) must match."
                )

            if init_len != n_clusters:
                raise ValueError(
                    f"The number of elements in the initial quantized distribution ({init_len}) should match the "
                    f"given number of optimal quants ({n_clusters})."
                )

            return init.copy()
        case "sample":
            random_indices = random_state.choice(
                x_len, size=n_clusters, replace=False
            )
            return x[random_indices]
        case "random":
            return random_state.rand(n_clusters, x_dims)
        case "milp":
            return _milp(x, n_clusters)
        case "k-means++" | None:
            return _kmeans_plus_plus(
                x, n_clusters, random_state
            )
        case _:
            raise ValueError(
                f"Initialization strategy ‘{init}’ is not a valid option. Supported options are "
                "{‘sample’, ‘random’, ‘k-means++’, ‘milp’}."
            )



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
    """Initializes cluster centers {yₖ} through the solution of a mixed-integer linear programming (MILP) problem with
    a cardinality constraint that minimizes the Kantorovich–Rubinstein distance between elements of the set {ξᵢ}:

    min(δᵢ,xᵢⱼ) Σⁱ₌₁ᵐ Σʲ₌₁ᵐ dist(ξᵢ,ξⱼ)xᵢⱼ

    Σᵢ₌₁ᵐ xᵢⱼ = qⱼ,   Σⱼ₌₁ᵐ xᵢⱼ ≤ δᵢ,   Σᵢ₌₁ᵐ δᵢ ≤ k

    i = 1, …, m    j = 1, …, m    δᵢ ∈ {0, 1}    xᵢⱼ ≥ 0

    With the location terminology, this problem finds locations of points ξᵢ and transportation volumes xᵢⱼ to consumer
    with demands qᵢ. A feasible solution is characterized by k selected elements ξᵢ, where selection variables δᵢ equal
    1. These selected variables constitute the optimal centers {yₖ}.

    Regarding the initialization strategy, the following parameters are defined:

     * {qᵢ} represents the selection probabilities of corresponding elements {ξᵢ}

     * dist(ξᵢ,ξⱼ) denotes the distance between elements ξᵢ and ξⱼ, expressed as: dist(ξᵢ, ξⱼ) = √Σₗ₌₁ⁿ (ξᵢₗ - ξⱼₗ)²

    Parameters
    ----------
    X : np.ndarray
        The input tensor containing training element {ξᵢ}.

    n_clusters : int or np.uint, default=2
        The number of initial cluster centers {yₖ}. Must be greater than or equal to 1.

    weights : np.ndarray, optional
        The selection probabilities (weights) {qᵢ} of corresponding elements {ξᵢ}. If not specified, the weights are
        set as a normalized unit vector.

    kwargs : dict, optional
        Additional keyword arguments passed to `scipy.optimize.milp` solver.

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

    ValueError
        If the selection probabilities vector {qᵢ} does not have one dimension or if the number of elements does not
        correspond to the number of elements in the input tensor {ξᵢ}.
    """

    X_len, _ = X.shape

    if not X_len:
        raise ValueError("The input tensor X should not be empty.")

    if n_clusters < 1:
        raise ValueError(
            "The number of initial cluster centers should not be less than 1."
        )

    if weights is None:
        # Assign default weights as a normalized unit vector
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

    # Separating selection variables from transportation volumes
    optimal_selection = result.x[-X_len:]
    center_indexes = np.argwhere(optimal_selection >= 0.5).flatten().tolist()
    cluster_centers = X[center_indexes, :]

    return cluster_centers
