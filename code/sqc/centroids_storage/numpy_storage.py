import numpy as np
from sklearn.exceptions import NotFittedError

from .factory import CentroidStorageFactory, CentroidStorage
from .init import init_centroids


@CentroidStorageFactory.register()
class NumpyCentroidStorage(CentroidStorage):
    name = "numpy"

    def __init__(
        self, n_clusters: int, init: str | np.ndarray = "k-means++", *args, **kwargs
    ):
        """
        Initializes the NumpyCentroidStorage class.

        Parameters
        ----------
        n_clusters : int
            The number of clusters (centroids) to initialize.
        init : str or np.ndarray, optional
            Method for initialization. Can be 'k-means++' or an ndarray of initial centroids.
        """
        super().__init__(n_clusters, init, *args, **kwargs)
        self._centroids = None

    @property
    def centroids(self) -> np.ndarray:
        if self._centroids is None:
            raise NotFittedError("Centroids have not been initialized yet.")
        return self._centroids

    def init_centroids(self, x: np.ndarray, random_state: np.random.RandomState):
        self._centroids = init_centroids(self._init, self._n_clusters, x, random_state)

    def find_nearest_centroid(self, target: np.ndarray) -> tuple[np.ndarray, np.uint]:
        y = self.centroids

        if y.size == 0 or target.size == 0:
            raise ValueError(
                "Either the `y` input tensor or the `target` tensor is empty."
            )

        if y.shape[1:] != target.shape:
            raise ValueError(
                "The dimensions of individual elements in `y` and `target` must match. Elements in `y` have "
                f"shape {y.shape[1:]}, but `target` tensor has shape {target.shape}."
            )

        distance = np.linalg.norm(target - y, axis=1)
        nearest_index = np.argmin(distance)

        return y[nearest_index, :], nearest_index

    def update_centroid(self, index: np.uint, delta: np.ndarray):
        self._centroids[index] -= delta


@CentroidStorageFactory.register(requires_filepath=True)
class NumpyMemmapCentroidStorage(NumpyCentroidStorage):
    name = "numpy_memmap"

    def __init__(
        self,
        filepath: str,
        n_clusters: int,
        init: str | np.ndarray = "k-means++",
        *args,
        **kwargs,
    ):
        """
        Initializes the NumpyCentroidStorage class.

        Parameters
        ----------
        n_clusters : int
            The number of clusters (centroids) to initialize.
        init : str or np.ndarray, optional
            Method for initialization. Can be 'k-means++' or an ndarray of initial centroids.
        """
        super().__init__(n_clusters, init, *args, **kwargs)
        self._filepath = filepath

    def init_centroids(self, x: np.ndarray, random_state: np.random.RandomState):
        _, x_dims = x.shape
        self._centroids = np.memmap(
            self._filepath, dtype=x.dtype, mode="w+", shape=(self._n_clusters, x_dims)
        )
        self._centroids[:] = init_centroids(
            self._init, self._n_clusters, x, random_state
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
