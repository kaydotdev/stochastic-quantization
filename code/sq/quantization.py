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
        init: Union[
            StochasticQuantizationInit, np.ndarray
        ] = StochasticQuantizationInit.K_MEANS_PLUS_PLUS,
        learning_rate: Union[float, np.float64] = 0.001,
        rank: Union[int, np.uint] = 3,
        tol: Optional[Union[float, np.float64]] = None,
        log_step: Optional[Union[int, np.uint]] = None,
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
        init : StochasticQuantizationInit or np.ndarray, default=StochasticQuantizationInit.K_MEANS_PLUS_PLUS
            Initialization strategy for {yₖ} elements in quantized distribution. Optionally, elements {yₖ} can be
            provided manually.
        learning_rate : float or np.float64, default=0.001
            The learning rate parameter ρ, which determines the convergence speed and stability of the algorithm. Must
            be greater than 0.
        rank : int or np.uint, default=3
            The degree of the norm (rank) r. Must be greater than or equal to 3.
        tol : float or np.float64, optional
            Relative tolerance with regard to objective function difference of two consecutive iterations to declare
            convergence. If not specified, the algorithm will run for 'max_iter' iterations.
        log_step : int or np.uint, optional
             The iteration interval for calculating and recording objective function value. If in verbose mode, the
             value is printed to STDOUT. If not specified, the logging step is set to the size of the input tensor {ξᵢ}.
        random_state : np.random.RandomState, optional
            Random state for reproducibility.
        verbose : int or np.uint, default=0
            Verbosity level: 0 for silent, 1 for progress logging to STDOUT.
        """

        self._optim = optim
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._init = init
        self._learning_rate = learning_rate
        self._rank = rank
        self._tol = tol
        self._log_step = log_step
        self._random_state = random_state
        self._verbose = verbose

    def fit(self, X: np.ndarray, y=None):
        """Search optimal values of {yₖ} using numeric iterative sequence, that updates parameters {yₖ} based on the
        calculated gradient value of a norm between sampled ξᵢ and the nearest element yₖ:

            1. k⁽ᵗ⁾ ∈ S(ξ̃⁽ᵗ⁾,y⁽ᵗ⁾) = argmin₁≤k≤K d(ξ̃⁽ᵗ⁾, yₖ⁽ᵗ⁾), t=0,1,…;
            2. gₖ⁽ᵗ⁾ = { r ‖ ξ̃⁽ᵗ⁾ - yₖ⁽ᵗ⁾ ‖ʳ⁻² (yₖ⁽ᵗ⁾ - ξ̃⁽ᵗ⁾), if k = k⁽ᵗ⁾; 0, if k ≠ k⁽ᵗ⁾;
            3. yₖ⁽ᵗ⁺¹⁾ = πY(yₖ⁽ᵗ⁾ - ρₜgₖ⁽ᵗ⁾), k=1,…,K;

        Parameters
        ----------
        X : np.ndarray
            The input tensor containing training element {ξᵢ}.
        y : None
            Ignored. This parameter exists only for compatibility with estimator interface.

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If the input tensor {ξᵢ} does not contain any elements.
        ValueError
            If dimensions of initial quantized distribution {y₀} and input tensor {ξᵢ} do not match.
        ValueError
            The number of elements in the initial quantized distribution {y₀} does not match the number of optimal
            quants in the class constructor.
        """

        random_state = check_random_state(self._random_state)
        X_len, X_dims = X.shape

        if not X_len:
            raise ValueError("The input tensor X should not be empty.")

        match self._init:
            case _ if isinstance(self._init, np.ndarray):
                init_len, init_dims = self._init.shape

                if init_dims != X_dims:
                    raise ValueError(
                        f"The dimensions of initial quantized distribution ({init_len}) and input tensor "
                        f"({X_dims}) must match."
                    )

                if init_len != self._n_clusters:
                    raise ValueError(
                        f"The number of elements in the initial quantized distribution ({init_len}) should match the "
                        f"given number of optimal quants ({self._n_clusters})."
                    )

                self.cluster_centers_ = self._init.copy()
            case StochasticQuantizationInit.SAMPLE:
                random_indices = random_state.choice(
                    X_len, size=self._n_clusters, replace=False
                )
                self.cluster_centers_ = X[random_indices]
            case StochasticQuantizationInit.RANDOM:
                self.cluster_centers_ = random_state.rand(self._n_clusters, X_dims)
            case StochasticQuantizationInit.K_MEANS_PLUS_PLUS:
                random_indices = random_state.choice(X_len, size=1, replace=False)
                self.cluster_centers_ = np.expand_dims(X[random_indices.item()], axis=0)

                for _ in range(1, self._n_clusters):
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

        if self._verbose:
            print("Initialization complete")

        self.n_iter_ = 0
        self.n_step_ = 0
        self._log_step = self._log_step or X_len
        self.loss_history_ = [calculate_loss(X, self.cluster_centers_)]
        self._optim.reset()

        for i in range(self._max_iter):
            self.n_iter_ += 1

            for ksi_j in np.random.permutation(X):
                self.n_step_ += 1

                nearest_quant, quant_ind = find_nearest_element(
                    self.cluster_centers_, ksi_j
                )

                grad_fn = (
                    lambda x: self._rank
                    * np.linalg.norm(ksi_j - x, ord=2) ** (self._rank - 2)
                    * (x - ksi_j)
                )

                self.cluster_centers_[quant_ind, :] = self._optim.step(
                    grad_fn, nearest_quant, self._learning_rate
                )

                if not self.n_step_ % self._log_step:
                    current_loss = calculate_loss(X, self.cluster_centers_)

                    self.loss_history_.append(current_loss)

                    if self._verbose:
                        print(
                            f"Gradient step [{self.n_step_}/{self._max_iter * X_len}]: loss={current_loss}"
                        )

            current_loss = calculate_loss(X, self.cluster_centers_)

            if (
                self._tol is not None
                and self.loss_history_[-1] - current_loss < self._tol
            ):
                if self._verbose:
                    print(
                        f"Converged (small optimal quants change) at step [{self.n_step_}/{self._max_iter * X_len}]"
                    )
                break

        return self

    def predict(self, X: np.ndarray):
        """Predict the closest optimal quant {yₖ} each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray
            New data to predict.

        Returns
        -------
        labels : np.ndarray
            Index of the optimal quant each sample belongs to.

        Raises
        ------
        NotFittedError
            If the attributes are not found.
        """

        check_is_fitted(self)

        pairwise_distance = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_, axis=-1
        )

        return np.argmin(pairwise_distance, axis=-1)
