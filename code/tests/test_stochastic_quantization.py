import unittest
import numpy as np

from sklearn.exceptions import NotFittedError

from sq.optim import SGDOptimizer
from sq.quantization import StochasticQuantization, StochasticQuantizationInit


class TestStochasticQuantization(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(seed=42)
        self.algorithm = StochasticQuantization(
            SGDOptimizer(), n_clusters=2, max_iter=1, random_state=self.random_state
        )
        self.X = np.array(
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

    def test_should_raise_not_fitted_error_if_fit_is_not_called(self):
        # arrange
        X = self.random_state.random((10, 3))

        # assert
        with self.assertRaises(NotFittedError):
            # act
            self.algorithm.predict(X)

    def test_should_raise_value_error_if_input_tensor_is_empty(self):
        # arrange
        X = np.array([])

        # assert
        with self.assertRaises(ValueError):
            # act
            self.algorithm.fit(X)

    def test_should_raise_value_error_if_initial_distribution_size_and_cluster_number_do_not_match(
        self,
    ):
        # arrange
        self.algorithm = StochasticQuantization(
            SGDOptimizer(),
            n_clusters=10,
            max_iter=1,
            random_state=self.random_state,
            init=np.array([[[0.0, 0.0], [0.0, 0.0]]]),
        )

        # assert
        with self.assertRaises(ValueError):
            # act
            self.algorithm.fit(self.X)

    def test_should_raise_value_error_if_dimensions_of_quantized_distribution_and_input_tensor_does_not_match(
        self,
    ):
        # arrange
        self.algorithm = StochasticQuantization(
            SGDOptimizer(),
            n_clusters=2,
            max_iter=1,
            random_state=self.random_state,
            init=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        )

        # assert
        with self.assertRaises(ValueError):
            # act
            self.algorithm.fit(self.X)

    def test_should_return_init_quants_if_input_tensor_contains_single_element_with_kmeans_plus_plus_init(
        self,
    ):
        # arrange
        self.algorithm = StochasticQuantization(
            SGDOptimizer(),
            n_clusters=1,
            max_iter=1,
            random_state=self.random_state,
            init=StochasticQuantizationInit.K_MEANS_PLUS_PLUS,
        )

        X = np.array(
            [
                [0.0, 0.0, 0.0],
            ]
        )

        # act
        self.algorithm.fit(X)

        # assert
        np.testing.assert_array_equal(self.algorithm.cluster_centers_, X, strict=True)
        self.assertEqual(self.algorithm.n_iter_, 1)

    def test_should_return_init_quants_if_max_iteration_is_zero_with_kmeans_plus_plus_init(
        self,
    ):
        # arrange
        self.algorithm = StochasticQuantization(
            SGDOptimizer(),
            n_clusters=2,
            max_iter=0,
            random_state=self.random_state,
            init=StochasticQuantizationInit.K_MEANS_PLUS_PLUS,
        )

        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        expected_cluster_centers = np.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )

        # act
        self.algorithm.fit(X)

        # assert
        np.testing.assert_array_equal(
            self.algorithm.cluster_centers_, expected_cluster_centers, strict=True
        )
        self.assertEqual(self.algorithm.n_iter_, 0)

    def test_should_return_optimal_quants_for_sampling_strategy(self):
        # arrange
        self.algorithm = StochasticQuantization(
            SGDOptimizer(),
            n_clusters=2,
            max_iter=1,
            random_state=self.random_state,
            init=StochasticQuantizationInit.SAMPLE,
        )

        expected_cluster_centers = np.array(
            [
                [0.679370, 0.440726],
                [0.832443, 0.212339],
            ]
        )

        # act
        self.algorithm.fit(self.X)

        # assert
        np.testing.assert_allclose(
            self.algorithm.cluster_centers_,
            expected_cluster_centers,
            rtol=1e-3,
            atol=1e-3,
        )
        self.assertAlmostEqual(self.algorithm.loss_history_[0], 3.47277, places=5)
        self.assertEqual(self.algorithm.n_iter_, 1)

    def test_should_return_optimal_quants_for_uniformly_distributed_quants_strategy(
        self,
    ):
        # arrange
        self.algorithm = StochasticQuantization(
            SGDOptimizer(),
            n_clusters=2,
            max_iter=1,
            random_state=self.random_state,
            init=StochasticQuantizationInit.RANDOM,
        )

        expected_cluster_centers = np.array(
            [
                [0.374253, 0.950713],
                [0.728225, 0.594985],
            ]
        )

        # act
        self.algorithm.fit(self.X)

        # assert
        np.testing.assert_allclose(
            self.algorithm.cluster_centers_,
            expected_cluster_centers,
            rtol=1e-3,
            atol=1e-3,
        )
        self.assertAlmostEqual(self.algorithm.loss_history_[0], 3.44586, places=5)
        self.assertEqual(self.algorithm.n_iter_, 1)

    def test_should_return_optimal_quants_for_kmeans_plus_plus_strategy(
        self,
    ):
        # arrange
        self.algorithm = StochasticQuantization(
            SGDOptimizer(),
            n_clusters=2,
            max_iter=1,
            random_state=self.random_state,
            init=StochasticQuantizationInit.K_MEANS_PLUS_PLUS,
        )

        expected_cluster_centers = np.array(
            [
                [0.681516, 0.438416],
                [0.065339, 0.948887],
            ]
        )

        # act
        self.algorithm.fit(self.X)

        # assert
        np.testing.assert_allclose(
            self.algorithm.cluster_centers_,
            expected_cluster_centers,
            rtol=1e-3,
            atol=1e-3,
        )
        self.assertAlmostEqual(self.algorithm.loss_history_[0], 2.65553, places=5)
        self.assertEqual(self.algorithm.n_iter_, 1)
