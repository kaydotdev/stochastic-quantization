from enum import Enum
from typing import Self, Optional

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from .metric import calculate_loss, find_nearest_element
from .optim import BaseOptimizer


class StochasticQuantizationInit(Enum):
    SAMPLE = "SAMPLE"
    RANDOM = "RANDOM"


class StochasticQuantization(BaseEstimator, ClusterMixin):
    """ """

    def __init__(
        self,
        optim: BaseOptimizer,
        *,
        n_clusters: int = 2,
        max_iter: int = 10,
        init: StochasticQuantizationInit = StochasticQuantizationInit.SAMPLE,
        learning_rate: np.float64 = 0.001,
        rank: np.unsignedinteger = 3,
        tol: Optional[np.float64] = None,
        random_state: Optional[np.random.RandomState] = None,
        verbose: int = 0,
    ):
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

    def fit(self, X: np.ndarray, y=None) -> Self:
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """

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
        """

        Parameters
        ----------
        X

        Returns
        -------

        """

        check_is_fitted(self)

        pairwise_distance = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_, axis=-1
        )

        return np.argmin(pairwise_distance, axis=-1)
