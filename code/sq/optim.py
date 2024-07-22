from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np


class BaseOptimizer(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError("State reset is available for subclasses only.")

    @abstractmethod
    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Parameters update calculation is available for subclasses only."
        )


class SGDOptimizer(BaseOptimizer):
    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        return x - learning_rate * grad_fn(x)

    def reset(self) -> None:
        pass


class MomentumOptimizer(BaseOptimizer):
    def __init__(self, gamma: np.float64 = 0.9):
        self.momentum_term = None
        self.gamma = gamma

    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        if self.momentum_term is None:
            self.momentum_term = np.zeros(shape=(1, x.size))

        self.momentum_term = self.gamma * self.momentum_term + learning_rate * grad_fn(
            x
        )

        return x - self.momentum_term

    def reset(self) -> None:
        self.momentum_term = None


class NAGOptimizer(BaseOptimizer):
    def __init__(self, gamma: np.float64 = 0.9):
        self.momentum_term = None
        self.gamma = gamma

    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        if self.momentum_term is None:
            self.momentum_term = np.zeros(shape=(1, x.size))

        self.momentum_term = self.gamma * self.momentum_term + learning_rate * grad_fn(
            x - self.gamma * self.momentum_term
        )

        return x - self.momentum_term

    def reset(self) -> None:
        self.momentum_term = None


class AdagradOptimizer(BaseOptimizer):
    def __init__(self, var_eps: np.float64 = 1e-8):
        self.grad_term = None
        self.var_eps = var_eps

    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        if self.grad_term is None:
            self.grad_term = np.zeros(shape=(1, x.size))

        grad_x = grad_fn(x)

        self.grad_term += grad_x**2

        return x - (learning_rate / np.sqrt(self.grad_term + self.var_eps)) * grad_x

    def reset(self) -> None:
        self.grad_term = None


class RMSPropOptimizer(BaseOptimizer):
    def __init__(self, beta: np.float64 = 0.9, var_eps: np.float64 = 1e-8):
        self.grad_term = None
        self.beta = beta
        self.var_eps = var_eps

    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        if self.grad_term is None:
            self.grad_term = np.zeros(shape=(1, x.size))

        grad_x = grad_fn(x)

        self.grad_term = self.beta * self.grad_term + (1 - self.beta) * grad_x**2

        return x - (learning_rate / np.sqrt(self.grad_term + self.var_eps)) * grad_x

    def reset(self) -> None:
        self.grad_term = None


class AdamOptimizer(BaseOptimizer):
    def __init__(
        self,
        betas: Tuple[np.float64, np.float64] = (0.9, 0.999),
        var_eps: np.float64 = 1e-8,
    ):
        self.momentum_term = None
        self.variance_term = None
        self.betas = betas
        self.var_eps = var_eps

    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        beta1, beta2 = self.betas

        if self.momentum_term is None and self.variance_term is None:
            self.momentum_term = np.zeros(shape=(1, x.size))
            self.variance_term = np.zeros(shape=(1, x.size))

        grad_x = grad_fn(x)

        self.momentum_term = beta1 * self.momentum_term + (1 - beta1) * grad_x
        self.variance_term = beta2 * self.variance_term + (1 - beta2) * grad_x**2

        return (
            x
            - (learning_rate / np.sqrt(self.variance_term + self.var_eps))
            * self.momentum_term
        )

    def reset(self) -> None:
        self.momentum_term = None
        self.variance_term = None
