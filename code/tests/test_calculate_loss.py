import unittest
import numpy as np

from sqg.centroids_storage.factory import CentroidStorageFactory


class TestCalculateLoss(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(seed=42)

    def test_should_raise_value_error_if_shape_mismatch(self):
        # arrange
        x = self.random_state.random((10, 2)) 
        y = self.random_state.random((10, 3))
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=10, init=y)

        # assert
        with self.assertRaises(ValueError):
            # act
            storage.init_centroids(x, self.random_state)
        cleanup()

    def test_should_raise_value_error_if_different_axis(self):
        # arrange
        x = self.random_state.random((1, 2))
        y = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=1, init=y)

        # assert
        with self.assertRaises(ValueError):
            # act
            storage.init_centroids(x, self.random_state)
        cleanup()

    def test_should_raise_value_error_if_one_of_distributions_is_empty(self):
        # arrange
        x = self.random_state.random((1, 2))
        y = np.array([])
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=1, init=y)

        # assert
        with self.assertRaises(ValueError):
            # act
            storage.calculate_loss(x)
        cleanup()

    def test_should_return_distance_for_distributions_with_different_size(self):
        # arrange
        expected_distance = 4.141985  # Updated for 2D data

        x = np.array(
            [
                [0.37454012, 0.95071431],
                [0.15601864, 0.15599452],
                [0.60111501, 0.70807258],
                [0.83244264, 0.21233911],
                [0.30424224, 0.52475643],
                [0.61185289, 0.13949386],
                [0.45606998, 0.78517596],
                [0.59241457, 0.04645041],
                [0.06505159, 0.94888554],
                [0.30461377, 0.09767211],
            ]
        )
        y = np.array(
            [
                [0.12203823, 0.49517691],
                [0.25877998, 0.66252228],
            ]
        )
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=2, init=y)
        storage.init_centroids(x, self.random_state)

        # act
        actual_distance = storage.calculate_loss(x)

        # assert
        self.assertAlmostEqual(actual_distance, expected_distance, places=6)
        cleanup()

    def test_should_return_zero_distance_for_identical_tensors(self):
        # arrange
        expected_distance = 0.0

        x = np.array([[1.0, 1.0], [1.0, 1.0]])
        y = np.array([[1.0, 1.0], [1.0, 1.0]])
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=2, init=y)
        storage.init_centroids(x, self.random_state)

        # act
        actual_distance = storage.calculate_loss(x)

        # assert
        self.assertAlmostEqual(actual_distance, expected_distance, places=7)
        cleanup()


if __name__ == "__main__":
    unittest.main()
