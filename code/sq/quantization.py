from typing import Optional, Union

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from .init import _kmeans_plus_plus, _milp
from .metric import _calculate_loss, _find_nearest_element
from .optim import BaseOptimizer


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
        learning_rate: Union[float, np.float64] = 0.001,
        rank: Union[int, np.uint] = 3,
        verbose: Union[int, np.uint] = 0,
        element_selection_method: Optional[str] = None,
        init: Optional[Union[str, np.ndarray]] = None,
        tol: Optional[Union[float, np.float64]] = None,
        log_step: Optional[Union[int, np.uint]] = None,
        random_state: Optional[np.random.RandomState] = None,
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

        learning_rate : float or np.float64, default=0.001
            The learning rate parameter ρ, which determines the convergence speed and stability of the algorithm. Must
            be greater than 0.

        rank : int or np.uint, default=3
            The degree of the norm (rank) r. Must be greater than or equal to 3.

        verbose : int or np.uint, default=0
            Verbosity mode (0 - silent mode, 1 - logs progress to STDOUT).

        element_selection_method : {‘permutation’, ‘sample’}, optional
            Method used to select elements uniformly from {ξᵢ} during each iteration:

            * 'permutation': Each element from {ξᵢ} is selected only once.

            * 'sample': Elements from {ξᵢ} can be selected multiple times.

            If not specified, defaults to 'permutation'.

        init : {‘sample’, ‘random’, ‘k-means++’, ‘milp’} or np.ndarray, optional
            Initialization strategy for the elements {yₖ} in the quantized distribution:

            * np.ndarray: Use the provided array as initial elements. Must have shape (n_clusters, n_features).

            * 'sample': Initialize elements by uniform sampling from {ξᵢ}.

            * 'random': Initialize elements by random sampling from a uniform distribution over [0, 1).

            * 'k-means++': Initialize elements using an empirical probability distribution based on point contributions
            to inertia for selecting initial centroids.

            * 'milp': Initialize elements through the solution of a mixed-integer linear programming (MILP) problem
            with a cardinality constraint.

            If not specified, defaults to 'k-means++'.

        tol : float or np.float64, optional
            Relative tolerance with regard to objective function difference of two consecutive iterations to declare
            convergence. If not specified, the algorithm will run for 'max_iter' iterations.

        log_step : int or np.uint, optional
             The iteration interval for calculating and recording objective function value. If in verbose mode, the
             value is printed to STDOUT. If not specified, the logging step is set to the size of the input tensor {ξᵢ}.

        random_state : np.random.RandomState, optional
            Random state for reproducibility.
        """

        self._optim = optim
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._element_selection_method = element_selection_method
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

        * k⁽ᵗ⁾ ∈ S(ξ̃⁽ᵗ⁾,y⁽ᵗ⁾) = argmin₁≤k≤K d(ξ̃⁽ᵗ⁾, yₖ⁽ᵗ⁾), t=0,1,…;

        * gₖ⁽ᵗ⁾ = { r ‖ ξ̃⁽ᵗ⁾ - yₖ⁽ᵗ⁾ ‖ʳ⁻² (yₖ⁽ᵗ⁾ - ξ̃⁽ᵗ⁾), if k = k⁽ᵗ⁾; 0, if k ≠ k⁽ᵗ⁾;

        * yₖ⁽ᵗ⁺¹⁾ = πY(yₖ⁽ᵗ⁾ - ρₜgₖ⁽ᵗ⁾), k=1,…,K;

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
            case "sample":
                random_indices = random_state.choice(
                    X_len, size=self._n_clusters, replace=False
                )
                self.cluster_centers_ = X[random_indices]
            case "random":
                self.cluster_centers_ = random_state.rand(self._n_clusters, X_dims)
            case "milp":
                self.cluster_centers_ = _milp(X, self._n_clusters)
            case "k-means++" | None:
                self.cluster_centers_ = _kmeans_plus_plus(
                    X, self._n_clusters, random_state
                )
            case _:
                raise ValueError(
                    f"Initialization strategy ‘{self._init}’ is not a valid option. Supported options are "
                    "{‘sample’, ‘random’, ‘k-means++’}."
                )

        if self._verbose:
            print("Initialization complete")

        self.n_iter_ = 0
        self.n_step_ = 0
        self._log_step = self._log_step or X_len
        self.loss_history_ = [_calculate_loss(X, self.cluster_centers_)]
        self._optim.reset()

        for i in range(self._max_iter):
            self.n_iter_ += 1

            match self._element_selection_method:
                case "permutation" | None:
                    ksi = (ksi_j for ksi_j in random_state.permutation(X))
                case "sample":
                    ksi = (
                        X[j]
                        for j in random_state.choice(X_len, size=X_len, replace=True)
                    )
                case _:
                    raise ValueError(
                        f"Element selection method ‘{self._element_selection_method}’ is not a valid option. Supported "
                        "options are {‘permutation’, ‘sample’}."
                    )

            for ksi_j in ksi:
                self.n_step_ += 1

                nearest_quant, quant_ind = _find_nearest_element(
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
                    current_loss = _calculate_loss(X, self.cluster_centers_)

                    self.loss_history_.append(current_loss)

                    if self._verbose:
                        print(
                            f"Gradient step [{self.n_step_}/{self._max_iter * X_len}]: loss={current_loss}"
                        )

            current_loss = _calculate_loss(X, self.cluster_centers_)

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
