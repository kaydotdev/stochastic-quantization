from typing import Optional, Union, Callable

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from .centroids_storage.factory import (
    CentroidStorage,
    StorageBackendType,
    CentroidStorageFactory,
)
from .progress_tracking import tqdm_joblib, tqdm
from .optim import BaseOptimizer
from .utils import batched


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
        backend: StorageBackendType = "numpy",
        element_selection_method: Optional[str] = None,
        init: Optional[Union[str, np.ndarray]] = None,
        tol: Optional[Union[float, np.float64]] = None,
        log_step: Optional[Union[int, np.uint]] = None,
        random_state: Optional[np.random.RandomState] = None,
        backend_kwargs: dict = None,
        **kwargs,
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
            Verbosity mode (0 - silent mode, 1 - progress input with tqdm, 2 - additional log info like loss).
            tqdm progress can be turned off by setting `use_tqdm=False` in `kwargs`. And loss is only printed when
            `log_step` is set.

        backend : StorageBackendType, default='numpy'
            The backend storage type for the centroid storage. Supported options are 'numpy' and 'numpy_memmap',
            'faiss' or any other CommandStorage implementation. Faiss backend requires the faiss library to be
            installed.

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

        backend_kwargs : dict, optional
            Additional keyword arguments for the centroid storage backend.
            `keep_filepath` - does not remove filepath on model cleanup

        Notes
        -----
        """

        self.n_iter_ = 0
        self.n_step_ = 0
        self._optim = optim
        self._max_iter = max_iter
        self._element_selection_method = element_selection_method
        self._learning_rate = learning_rate
        self._rank = rank
        self._tol = tol
        self._log_step = log_step
        self._random_state = random_state
        self._verbose = verbose
        self._verbose_details = verbose > 1
        self._verbose_progress = verbose > 0 and kwargs.get("use_tqdm", True)
        self.loss_history_ = []
        self.iteration_loss_history_ = []
        storage, deferred_action = CentroidStorageFactory.create(
            backend, n_clusters, init, **(backend_kwargs or {})
        )
        self._centroid_storage: CentroidStorage = storage
        self._deferred_action: Callable[[], None] = deferred_action
        self._kwargs = kwargs

    def __del__(self):
        self._deferred_action()

    @property
    def centroids(self) -> np.ndarray:
        return self._centroid_storage.centroids

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Returns the cluster centers (centroids).

        Returns
        -------
        np.ndarray
            The cluster centers.
        """
        return self._centroid_storage.centroids

    def _shuffle_ksi(self, X: np.ndarray, random_state: np.random.RandomState):
        """Shuffle the input tensor {ξᵢ} based on the specified element selection method.
        Parameters
        ----------
        X : np.ndarray
            The input tensor containing training element {ξᵢ}.
        random_state : np.random.RandomState
            Random state for reproducibility.
        Returns
        -------
        ksi : generator
            A generator that yields shuffled elements from the input tensor {ξᵢ}.
        """
        X_len, _ = X.shape
        match self._element_selection_method:
            case "permutation" | None:
                ksi = (ksi_j for ksi_j in random_state.permutation(X))
            case "sample":
                ksi = (
                    X[j] for j in random_state.choice(X_len, size=X_len, replace=True)
                )
            case _:
                raise ValueError(
                    f"Element selection method ‘{self._element_selection_method}’ is not a valid option. Supported "
                    "options are {‘permutation’, ‘sample’}."
                )
        return ksi

    def reset(self):
        """Reset the Stochastic Quantization solver to its initial state."""
        self.n_iter_ = 0
        self.n_step_ = 0
        self.loss_history_ = []
        self.iteration_loss_history_ = []
        self._optim.reset()

    def fit(self, X: np.ndarray, y=None, n_jobs: int = 1):
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
        n_jobs : int, default=1
            The number of jobs to run in parallel. If -1, use all processors.

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

        self._centroid_storage.init_centroids(X, random_state)
        if self._verbose_details:
            print("Initialization complete")

        self.reset()
        if self._log_step or self._tol:
            initial_loss = self._centroid_storage.calculate_loss(X)
            self.iteration_loss_history_.append(initial_loss)
            self.loss_history_.append(initial_loss)
            if self._verbose_details:
                print("Initial loss:", initial_loss)

        for i in range(self._max_iter):
            self.n_iter_ += 1

            ksi = self._shuffle_ksi(X, random_state)

            if n_jobs == 1:
                for ksi_j in tqdm(
                    ksi,
                    total=X_len,
                    desc="Performing cluster optimization",
                    disable=not self._verbose_progress,
                ):
                    self._optimize(
                        self._centroid_storage,
                        self._optim,
                        ksi_j,
                        self._rank,
                        self._learning_rate,
                    )
                    self.n_step_ += 1
                    self.__log_step(X, X_len)
            else:
                with tqdm_joblib(
                    total=X_len,
                    desc="Performing cluster optimization",
                    disable=not self._verbose_progress,
                ):
                    size = self._log_step or X_len
                    for ksi_batch in batched(ksi, size):
                        joblib.Parallel(
                            n_jobs=n_jobs,
                            max_nbytes=self._kwargs.get("joblib_max_nbytes", "50M"),
                        )(
                            joblib.delayed(self._optimize)(
                                self._centroid_storage,
                                self._optim,
                                ksi_j,
                                self._rank,
                                self._learning_rate,
                            )
                            for ksi_j in ksi_batch
                        )
                        self.n_step_ += size
                        self.__log_step(X, X_len)

            if self._tol is not None and self._early_stop(X):
                if self._verbose_details:
                    print(
                        f"Converged (small optimal quants change) at step [{self.n_step_}/{self._max_iter * X_len}] "
                        f"with loss={self.iteration_loss_history_[-1]} (iteration {self.n_iter_}, step "
                        f"{self.n_step_ % X_len})"
                    )
                break
        return self

    def __log_step(self, X: np.ndarray, X_len: int):
        """Log the objective function value at the specified step.
        Parameters
        ----------
        X : np.ndarray
            The input tensor containing training element {ξᵢ}.
        """
        if self._log_step and self.n_step_ % self._log_step == 0:
            current_loss = self._centroid_storage.calculate_loss(X)
            self.loss_history_.append(current_loss)

            if self._verbose_details:
                print(
                    f"Gradient step [{self.n_step_}/{self._max_iter * X_len}]: loss={current_loss} "
                    f"(iter: {self.n_iter_}, step: {self.n_step_ % X_len})"
                )

    def _early_stop(self, X: np.ndarray) -> bool:
        """Early stop the Stochastic Quantization solver
        based on the relative difference between the last two objective function values.
        Parameters
        ----------
        X : np.ndarray
            The input tensor containing training element {ξᵢ}.
        Returns
        -------
        bool
            True if the relative difference is less than the tolerance, False otherwise.
        """
        current_loss = self._centroid_storage.calculate_loss(X)
        self.iteration_loss_history_.append(current_loss)
        return self.iteration_loss_history_[-2] - current_loss < self._tol

    @staticmethod
    def _optimize(
        centroid_storage: CentroidStorage,
        optim: BaseOptimizer,
        ksi_j: np.array,
        rank: int,
        learning_rate: Union[float, np.float64],
    ):
        """Perform optimization step for a single sample.
        Parameters
        ----------
        centroid_storage : CentroidStorage
            Centroid storage object containing the current quantized distribution.
        optim : BaseOptimizer
            Optimizer object used for gradient descent.
        ksi_j : np.ndarray
            Sample from the input tensor {ξᵢ}.
        """
        nearest_quant, quant_ind = centroid_storage.find_nearest_centroid(ksi_j)

        grad_fn = (
            lambda x: rank
            * np.linalg.norm(ksi_j - x, ord=2) ** (rank - 2)
            * (x - ksi_j)
        )

        delta = optim.step(grad_fn, nearest_quant, learning_rate)
        centroid_storage.update_centroid(quant_ind, delta)

    def predict(self, X: np.ndarray, n_jobs: int = 1):
        """Predict the closest optimal quant {yₖ} each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray
            New data to predict.
        n_jobs : int, default=1
            The number of jobs to run in parallel. If -1, use all processors.

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

        _predict = lambda storage, target: storage.find_nearest_centroid(target)[1]

        if n_jobs == 1:
            clusters = [
                _predict(self._centroid_storage, target)
                for target in tqdm(
                    X,
                    desc="Prediction of the closet cluster",
                    disable=not self._verbose_progress,
                )
            ]
        else:
            with tqdm_joblib(
                total=len(X),
                desc="Prediction of the closet cluster",
                disable=not self._verbose_progress,
            ):
                clusters = joblib.Parallel(n_jobs=n_jobs)(
                    joblib.delayed(_predict)(self._centroid_storage, target)
                    for target in X
                )

        return clusters
