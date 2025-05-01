import numpy as np

from centroids_storage.factory import CentroidStorageFactory
from centroids_storage.numpy_storage import NumpyMemmapCentroidStorage


def _load_faiss():
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "Faiss is not installed. Please install it using extras `pip install sq[faiss]` or `pip install faiss-cpu`."
        ) from e
    return faiss


@CentroidStorageFactory.register(requires_filepath=True)
class FaissIndexBasedCentroidStorage(NumpyMemmapCentroidStorage):
    name = "faiss"

    def __init__(
            self,
            filepath: str,
            n_clusters: int,
            init: str | np.ndarray = "k-means++",
            voronoi_cell_size: int = 100,
            *args, **kwargs
    ):
        """
        FaissIndexBasedCentroidStorage is a centroid storage class that uses FAISS for efficient nearest neighbor
        search and centroid updates. It is designed to work with large datasets.

        Parameters
        ----------
        n_clusters : int
            The number of clusters (centroids) to initialize.
        init : str or np.ndarray, optional
            Method for initialization. Can be 'k-means++' or an ndarray of initial centroids.
        filepath : str or None, optional
            Path to the file where centroids are stored. If None, centroids are stored in memory.
        voronoi_cell_size : int, optional
            The size of the Voronoi cells for the FAISS index. Default is 100.
        """
        super().__init__(filepath, n_clusters, init, *args, **kwargs)
        _faiss = _load_faiss()
        self.index: _faiss.Index | None = None
        self._dim = None
        self._voronoi_cell_size = voronoi_cell_size
        self._faiss = _faiss

    def init_centroids(self, X: np.ndarray, random_state: np.random.RandomState):
        x_len, x_dims = X.shape
        self._dim = x_dims
        quantizer = self._faiss.IndexFlatL2(self._dim)
        index = self._faiss.IndexIVFFlat(quantizer, self._dim, self._voronoi_cell_size, self._faiss.METRIC_L2)
        super().init_centroids(X, random_state)
        index.train(self._centroids)
        index.add_with_ids(self._centroids, np.arange(0, self.n_clusters))
        self.index = index

    def find_nearest_centroid(self, target: np.ndarray) -> tuple[np.ndarray, np.uint]:
        distances, indices = self.index.search(np.atleast_2d(target), 1)
        nearest_index = indices[0, 0]
        nearest_centroid = self._centroids[nearest_index]
        return nearest_centroid, np.int64(nearest_index)

    def update_centroid(self, index: np.uint, delta: np.ndarray):
        value = self._centroids[index]
        index_array = np.array([index])
        self.index.remove_ids(index_array)
        updated_value = np.atleast_2d(value - delta)
        self._centroids[index] = updated_value
        self.index.add_with_ids(updated_value, index_array)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('_faiss', None)
        state['faiss_index'] = self._faiss.serialize_index(self.index)
        return state

    def __setstate__(self, state):
        self._faiss = _load_faiss()
        faiss_index = state.pop('faiss_index')
        super().__setstate__(state)
        self.index = self._faiss.deserialize_index(faiss_index)
