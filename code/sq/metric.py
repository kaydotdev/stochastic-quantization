from typing import Tuple

import numpy as np


def calculate_loss(x: np.ndarray, y: np.ndarray) -> np.float64:
    pairwise_distance = np.linalg.norm(x[:, np.newaxis] - y, axis=-1)
    min_distance = np.min(pairwise_distance, axis=-1)

    return np.sum(min_distance)


def find_nearest_element(
    x: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.signedinteger]:
    distance = np.linalg.norm(target - x, axis=1)
    nearest_index = np.argmin(distance)

    return x[nearest_index, :], nearest_index
