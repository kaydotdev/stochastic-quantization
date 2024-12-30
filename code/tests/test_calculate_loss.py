import unittest
import numpy as np


from sq.quantization import _calculate_loss


class TestCalculateLoss(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(seed=42)

    def test_should_raise_value_error_if_shape_mismatch(self):
        # arrange
        x = self.random_state.random((10, 2, 2))
        y = self.random_state.random((10, 1, 2))

        # assert
        with self.assertRaises(ValueError):
            # act
            _calculate_loss(x, y)

    def test_should_raise_value_error_if_different_axis(self):
        # arrange
        x = self.random_state.random((1, 2, 2))
        y = self.random_state.random((1, 2, 2, 2))

        # assert
        with self.assertRaises(ValueError):
            # act
            _calculate_loss(x, y)

    def test_should_raise_value_error_if_one_of_distributions_is_empty(self):
        # arrange
        x = self.random_state.random((1, 2, 2))
        y = np.array([])

        # assert
        with self.assertRaises(ValueError):
            # act
            _calculate_loss(x, y)

    def test_should_return_distance_for_distributions_with_different_size(self):
        # arrange
        expected_distance = 7.17435203

        x = np.array(
            [
                [[0.37454012, 0.95071431], [0.73199394, 0.59865848]],
                [[0.15601864, 0.15599452], [0.05808361, 0.86617615]],
                [[0.60111501, 0.70807258], [0.02058449, 0.96990985]],
                [[0.83244264, 0.21233911], [0.18182497, 0.18340451]],
                [[0.30424224, 0.52475643], [0.43194502, 0.29122914]],
                [[0.61185289, 0.13949386], [0.29214465, 0.36636184]],
                [[0.45606998, 0.78517596], [0.19967378, 0.51423444]],
                [[0.59241457, 0.04645041], [0.60754485, 0.17052412]],
                [[0.06505159, 0.94888554], [0.96563203, 0.80839735]],
                [[0.30461377, 0.09767211], [0.68423303, 0.44015249]],
            ]
        )
        y = np.array(
            [
                [[0.12203823, 0.49517691], [0.03438852, 0.9093204]],
                [[0.25877998, 0.66252228], [0.31171108, 0.52006802]],
            ]
        )

        # act
        actual_distance = _calculate_loss(x, y)

        # assert
        self.assertAlmostEqual(actual_distance, expected_distance, places=7)

    def test_should_return_zero_distance_for_identical_tensors(self):
        # arrange
        expected_distance = 0.0

        x = np.array([[1.0, 1.0], [1.0, 1.0]])
        y = np.array([[1.0, 1.0], [1.0, 1.0]])

        # act
        actual_distance = _calculate_loss(x, y)

        # assert
        self.assertAlmostEqual(actual_distance, expected_distance, places=7)


if __name__ == "__main__":
    unittest.main()
