import unittest
import numpy as np

from sqg.centroids_storage.factory import CentroidStorageFactory


class TestFindNearestElement(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(seed=42)

    def test_should_raise_value_error_if_shape_mismatch(self):
        # arrange
        x = self.random_state.random((10, 2))
        y = self.random_state.random((1, 3))
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=10, init=x)
        storage.init_centroids(x, self.random_state)

        # assert
        with self.assertRaises(ValueError):
            # act
            storage.find_nearest_centroid(y)
        cleanup()

    def test_should_raise_value_error_if_different_axis(self):
        # arrange
        x = self.random_state.random((1, 2))
        y = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # 3D array for init
        
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
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=1, init=x)
        storage.init_centroids(x, self.random_state)

        # assert
        with self.assertRaises(ValueError):
            # act
            storage.find_nearest_centroid(y)
        cleanup()

    def test_should_return_nearest_element_with_index(self):
        # arrange
        expected_element = np.array([0.15601864, 0.15599452])
        expected_index = 6

        x = np.array(
            [
                [0.30461377, 0.09767211],
                [0.68423303, 0.44015249],
                [0.06505159, 0.94888554],
                [0.29214465, 0.36636184],
                [0.18182497, 0.18340451],
                [0.83244264, 0.21233911],
                [0.15601864, 0.15599452],
                [0.37454012, 0.95071431],
            ]
        )
        y = np.array([0.0, 0.0])
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=8, init=x)
        storage.init_centroids(x, self.random_state)

        # act
        actual_element, actual_index = storage.find_nearest_centroid(y)

        # assert
        np.testing.assert_allclose(
            actual_element, expected_element, rtol=1e-3, atol=1e-3
        )
        self.assertEqual(actual_index, expected_index)
        cleanup()

    def test_should_return_first_index_of_multiple_nearest_elements(self):
        # arrange
        expected_element = np.array([0.15601864, 0.15599452])
        expected_index = 4

        x = np.array(
            [
                [0.30461377, 0.09767211],
                [0.68423303, 0.44015249],
                [0.06505159, 0.94888554],
                [0.29214465, 0.36636184],
                [0.15601864, 0.15599452],
                [0.15601864, 0.15599452],
                [0.15601864, 0.15599452],
                [0.37454012, 0.95071431],
            ]
        )
        y = np.array([0.0, 0.0])
        
        storage, cleanup = CentroidStorageFactory.create("numpy", n_clusters=8, init=x)
        storage.init_centroids(x, self.random_state)

        # act
        actual_element, actual_index = storage.find_nearest_centroid(y)

        # assert
        np.testing.assert_allclose(
            actual_element, expected_element, rtol=1e-3, atol=1e-3
        )
        self.assertEqual(actual_index, expected_index)
        cleanup()


if __name__ == "__main__":
    unittest.main()
