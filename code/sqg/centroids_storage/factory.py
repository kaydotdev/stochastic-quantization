import abc
import tempfile
from copy import deepcopy
from typing import Literal, Callable

import numpy as np


class CentroidStorage(abc.ABC):
    """
    Abstract base class for centroid storage implementations.

    Parameters
    ----------
    n_clusters : int
        The number of clusters (centroids) to initialize.
    init : str or np.ndarray, optional
        Method for initialization. Can be 'k-means++' or an ndarray of initial centroids.
    """

    def __init__(self, n_clusters: int, init: str | np.ndarray = "k-means++", *args, **kwargs):
        self._n_clusters = n_clusters
        self._init = init

    @property
    def n_clusters(self) -> int:
        """
        Returns the number of clusters.

        Returns
        -------
        int
            The number of clusters.
        """
        return self._n_clusters

    @property
    @abc.abstractmethod
    def centroids(self):
        """
        Returns the centroids.

        Returns
        -------
        np.ndarray
            The centroids.

        Raises
        ------
        ValueError
            If the centroids have not been initialized yet.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns the name of the centroid storage implementation.

        Returns
        -------
        str
            The name of the centroid storage implementation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_centroids(self, X: np.ndarray, random_state: np.random.RandomState):
        """
        Initializes the centroids.

        Parameters
        ----------
        X : np.ndarray
            The data to initialize the centroids.
        random_state : np.random.RandomState
            The random state for reproducibility.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def find_nearest_centroid(self, target: np.ndarray) -> tuple[np.ndarray, np.uint]:
        """
        Finds the nearest centroid to the target.

        Parameters
        ----------
        target : np.ndarray
            The target data point.

        Returns
        -------
        tuple[np.ndarray, np.uint]
            The nearest centroid and its index.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_centroid(self, index: np.uint, delta: np.ndarray):
        """
        Updates the centroid at the given index.

        Parameters
        ----------
        index : np.uint
            The index of the centroid to update.
        delta : np.ndarray
            The change to apply to the centroid.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def calculate_loss(self, X: np.ndarray) -> np.float64:
        """Calculates stochastic Wasserstein (or Kantorovich–Rubinstein) distance between distributions ξ and y:

        F(y) = Σᵢ₌₁ᴵ pᵢ min₁≤k≤K d(ξᵢ, yₖ)ʳ

        Parameters
        ----------
        xi : np.ndarray
            The original distribution ξ with shape (N, D, ...).
        y : np.ndarray
            The quantized distribution y with shape (M, D, ...).

        Returns
        -------
        np.float64
            The calculated stochastic Wasserstein distance between distributions ξ and y.

        Raises
        ------
        ValueError
            If one of the distributions ξ or y is empty.
        ValueError
            If there is a shape mismatch between individual elements in distribution ξ and y.

        Notes
        -----
        The function assumes uniform weights (pᵢ = 1) for all elements in the original distribution.
        The exponent r in the formula is implicitly set to 1 in this implementation.
        """

        if X.size == 0 or self.centroids.size == 0:
            raise ValueError("One of the distributions `X` or `centroids` is empty.")

        if X.shape[1:] != self.centroids.shape[1:]:
            raise ValueError(
                "The dimensions of individual elements in distribution `X` and `centroids` must match. Elements in "
                f"`X` have shape {X.shape[1:]}, but y elements have shape {self.centroids.shape[1:]}."
            )

        distances = [
            np.linalg.norm(self.find_nearest_centroid(ksi)[0] - ksi) for ksi in X
        ]

        return np.sum(distances)


StorageBackendType = Literal["numpy", "numpy_memmap", "faiss"] | CentroidStorage


class CentroidStorageFactory:
    """
    Factory class for creating centroid storage instances.
    """
    _implementations: dict[str, tuple[type[CentroidStorage], bool]] = dict()

    @classmethod
    def register(cls, requires_filepath: bool = False):
        """
        Registers a new centroid storage implementation.

        Parameters
        ----------
        storage_type : type[CentroidStorage]
            The centroid storage implementation to register.

        Returns
        -------
        type[CentroidStorage]
            The registered centroid storage implementation.
        """

        def decorator(storage_type: type[CentroidStorage]):
            cls._implementations[storage_type.name] = (storage_type, requires_filepath)
            return storage_type

        return decorator

    @classmethod
    def create(
            cls,
            storage_type: StorageBackendType,
            n_clusters: int,
            init: str | np.ndarray = "k-means++",
            **kwargs
    ) -> tuple[CentroidStorage, Callable[[], None]]:
        """
        Creates a centroid storage instance.

        Parameters
        ----------
        storage_type : StorageBackendType
            The type of storage backend to use.
        n_clusters : int
            The number of clusters (centroids) to initialize.
        init : str or np.ndarray, optional
            Method for initialization. Can be 'k-means++' or an ndarray of initial centroids.
        **kwargs
            Additional keyword arguments for the storage implementation.

        Returns
        -------
        CentroidStorage
            The created centroid storage instance.

        Raises
        ------
        ValueError
            If the storage type is unknown.
        """
        if isinstance(storage_type, CentroidStorage):
            return storage_type, lambda: None
        if isinstance(storage_type, str) and storage_type in cls._implementations:
            kwargs = deepcopy(kwargs)
            storage_implementation, requires_filepath = cls._implementations[storage_type]
            if requires_filepath and "filepath" not in kwargs:
                memory_file = tempfile.NamedTemporaryFile()
                kwargs["filepath"] = memory_file.name
                print(kwargs["filepath"])
            return storage_implementation(
                n_clusters=n_clusters, init=init, **kwargs
            ), (lambda: memory_file.close()) if not kwargs.get("keep_filepath") else lambda: None
        raise ValueError(
            f"Unknown storage type: {storage_type}, supported types are: {sorted(cls._implementations.keys())} "
            f"or {CentroidStorage.__name__} instance"
        )
