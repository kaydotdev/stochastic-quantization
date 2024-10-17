from enum import Enum
from typing import Optional, Union

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from .metric import calculate_loss, find_nearest_element
from .optim import BaseOptimizer


class StochasticQuantizationInit(Enum):
    SAMPLE = "SAMPLE"
    RANDOM = "RANDOM"
    K_MEANS_PLUS_PLUS = "K_MEANS_PLUS_PLUS"


class StochasticQuantization(BaseEstimator, ClusterMixin):
    """Solver for a Stochastic Quantization problem using a specified gradient search algorithm and initialization
    strategy. The algorithm finds the optimal quantized distribution by minimizing the Wasserstein (or
    Kantorovich–Rubinstein) distance between the input distribution and the quantized distribution. This distance
    serves as the objective function of the problem:

    min₍y = {y₁, …, yₖ} ∈ Y^K ⊂ ℝ^(nK)₎ F(y₁, …, yₖ)

    where:

    F(y) = F(y₁, …, yₖ) = Σᵢ₌₁ᴵ pᵢ min₁≤ₖ≤ₖ d(ξᵢ, yₖ)ʳ = 𝔼ᵢ∼ₚ min₁≤ₖ≤ₖ d(ξᵢ, yₖ)ʳ

    Attributes
    ----------
    loss_history_ : list
        History of objective function values corresponding to each iteration.
    n_iter_ : int
        Number of iterations until declaring convergence.
    cluster_centers_ : ndarray
        The optimal set of quantized points {y₁, …, yₖ}.
    """

    def __init__(
        self,
        optim: BaseOptimizer,
        *,
        n_clusters: Union[int, np.uint] = 2,
        max_iter: Union[int, np.uint] = 1,
        init: StochasticQuantizationInit = StochasticQuantizationInit.K_MEANS_PLUS_PLUS,
        learning_rate: Union[float, np.float64] = 0.001,
        rank: Union[int, np.uint] = 3,
        tol: Optional[Union[float, np.float64]] = None,
        random_state: Optional[np.random.RandomState] = None,
        verbose: Union[int, np.uint] = 0,
    ):
        """Initialize Stochastic Quantization solver with provided hyperparameters.

        Parameters
        ----------
        optim : BaseOptimizer
            Gradient search algorithm to use for finding optimal quantized distribution.
        n_clusters : int or np.uint, default=2
            The number of elements in tensor {yₖ} containing quantized distribution. Must be greater than or equal to 1.
        max_iter : int or np.uint, default=1
            Maximum number of iterations for the algorithm to converge. In a single iteration, the algorithm samples
            all elements from {ξᵢ} uniformly. Must be greater than or equal to 1.
        init : StochasticQuantizationInit, default=StochasticQuantizationInit.K_MEANS_PLUS_PLUS
            Initialization strategy for {yₖ} elements in quantized distribution.
        learning_rate : float or np.float64, default=0.001
            The learning rate parameter ρ, which determines the convergence speed and stability of the algorithm. Must
            be greater than 0.
        rank : int or np.uint, default=3
            The degree of the norm (rank) r. Must be greater than or equal to 3.
        tol : float or np.float64, optional
            Relative tolerance with regard to objective function difference of two consecutive iterations to declare
            convergence. If not specified, the algorithm will run for 'max_iter' iterations.
        random_state : np.random.RandomState, optional
            Random state for reproducibility.
        verbose : int or np.uint, default=0
            Verbosity mode.
        """

        self.loss_history_ = []
        self.n_iter_ = 0
        self.cluster_centers_ = np.array([])

        self.optim = optim
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.learning_rate = learning_rate
        self.rank = rank
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y=None):
        random_state = check_random_state(self.random_state)
        X_len, X_dims = X.shape

        match self.init:
            case StochasticQuantizationInit.SAMPLE:
                random_indices = random_state.choice(
                    X_len, size=self.n_clusters, replace=False
                )
                self.cluster_centers_ = X[random_indices]
            case StochasticQuantizationInit.RANDOM:
                self.cluster_centers_ = random_state.rand(self.n_clusters, X_dims)
            case StochasticQuantizationInit.K_MEANS_PLUS_PLUS:
                random_indices = random_state.choice(X_len, size=1, replace=False)
                self.cluster_centers_ = np.expand_dims(X[random_indices.item()], axis=0)

                for _ in range(1, self.n_clusters):
                    pairwise_distance = np.min(
                        np.linalg.norm(
                            X[:, np.newaxis] - self.cluster_centers_, axis=-1
                        ),
                        axis=-1,
                    )
                    pairwise_probabilities = pairwise_distance / np.sum(
                        pairwise_distance
                    )
                    cumulative_probabilities = np.cumsum(pairwise_probabilities)

                    next_centroid_index = np.searchsorted(
                        cumulative_probabilities, random_state.rand()
                    )
                    next_centroid = X[next_centroid_index]

                    self.cluster_centers_ = np.vstack(
                        (self.cluster_centers_, next_centroid)
                    )

        self.n_iter_ = 0
        self.loss_history_ = [calculate_loss(X, self.cluster_centers_)]
        self.optim.reset()

        for i in range(self.max_iter):
            for ksi_j in np.random.permutation(X):
                nearest_quant, quant_ind = find_nearest_element(
                    self.cluster_centers_, ksi_j
                )

                grad_fn = (
                    lambda x: self.rank
                    * np.linalg.norm(ksi_j - x, ord=2) ** (self.rank - 2)
                    * (x - ksi_j)
                )

                self.cluster_centers_[quant_ind, :] = self.optim.step(
                    grad_fn, nearest_quant, self.learning_rate
                )

            current_loss = calculate_loss(X, self.cluster_centers_)

            if (
                self.tol is not None
                and self.loss_history_[-1] - current_loss < self.tol
            ):
                break

            self.loss_history_.append(current_loss)
            self.n_iter_ += 1

        return self

    def predict(self, X: np.ndarray):
        check_is_fitted(self)

        pairwise_distance = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_, axis=-1
        )

        return np.argmin(pairwise_distance, axis=-1)
