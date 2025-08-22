from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for gradient descent optimization algorithms. This class defines
    the interface for optimization algorithms used in gradient descent. Subclasses must
    implement the `reset` and `step` methods to provide specific optimization behavior.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer's internal state to initial values.

        Subclasses must implement this method to clear any accumulated state variables used
        in the optimization process.

        Raises
        -------
        NotImplementedError
            If called directly on base class.
        """

        raise NotImplementedError("State reset is available for subclasses only.")

    @abstractmethod
    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        """Immutably compute and return updated parameter values using the gradient descent step.

        Parameters
        ----------
        grad_fn : Callable
            Objective function.
        x : np.ndarray
            Current parameter values as numpy tensor.
        learning_rate : np.float64
            The learning rate parameter ρ, which determines the convergence speed and stability of the algorithm. Must
            be greater than 0.

        Returns
        -------
        parameters : np.ndarray
            Update delta value.

        Raises
        -------
        NotImplementedError
            If called directly on base class.
        """

        raise NotImplementedError(
            "Parameters update calculation is available for subclasses only."
        )


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent (SGD) optimizer implementation.
    The optimizer is the simplest quasi-gradient algorithm defined
    as the recurrent update rule:

    xᵏ⁺¹ = Πₓ(xᵏ - ρₖ∇f(xᵏ)), Πₓ(y) = arg min_{x∈X} ‖y - x‖, x⁰∈X, k∈ℕ

    where:
        - k: iteration number
        - xᵏ: current point at iteration k
        - ∇f(xᵏ): gradient of the objective function at xᵏ
        - ρₖ: learning rate (step size) at iteration k
        - Πₓ: orthogonal projection operator onto set X

    Notes
    -----
        - The projection operator Πₓ(y) is currently implemented as the identity function: Πₓ(y) = y
        - This implementation extends BaseOptimizer for gradient-based optimization

    See Also
    --------
        BaseOptimizer : Parent class defining the optimizer interface
    """

    def step(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        learning_rate: np.float64,
    ) -> np.ndarray:
        return learning_rate * grad_fn(x)

    def reset(self) -> None:
        pass


class MomentumOptimizer(BaseOptimizer):
    """Momentum Gradient Descent (Heavy Ball Method) optimizer implementation.
    This optimizer extends standard Stochastic Gradient Descent (SGD) by
    incorporating a momentum term with coefficient 0 < γ < 1 (gamma). The method
    is inspired by the physics of motion under friction and follows the update rule:

    xᵏ⁺¹ = xᵏ + γₖ(xᵏ - xᵏ⁻¹) - ρₖ∇f(xᵏ), x⁰∈X, k∈ℕ

    where:
        - k: iteration number
        - xᵏ: parameters at iteration k
        - γₖ (gamma): momentum coefficient
        - ρₖ: learning rate
        - ∇f(xᵏ): gradient of the objective function at xᵏ

    Parameters
    ----------
    gamma : float64, optional
        Momentum coefficient, default is 0.9. Must be in range (0, 1).

    Attributes
    ----------
    momentum_term : ndarray or None
        Stores the momentum term for parameter updates.
    gamma : float64
        Momentum coefficient value.

    Notes
    -----
    The momentum term helps accelerate gradients in the relevant direction and
    dampens oscillations, often leading to faster convergence than standard SGD.
    """

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
            self.momentum_term = np.zeros(shape=x.size)

        self.momentum_term = self.gamma * self.momentum_term + learning_rate * grad_fn(
            x
        )

        return self.momentum_term

    def reset(self) -> None:
        self.momentum_term = None


class NAGOptimizer(BaseOptimizer):
    """Nesterov Accelerated Gradient (NAG) optimizer implementation.
    This optimizer extends momentum-based optimization by introducing
    an extrapolation step for more accurate parameter updates. NAG,
    also known as the ravine step method, improves upon the "Heavy Ball
    Method" by looking ahead in the optimization trajectory. The method
    follows the iterative process:

    xᵏ = yᵏ - ρₖ∇f(yᵏ), yᵏ⁺¹ = xᵏ + ρₖ(xᵏ - xᵏ⁻¹), x⁰ = y⁰∈X, k∈ℕ

    where:
        - k: iteration number
        - xᵏ: parameters at iteration k
        - yᵏ: extrapolated estimation of parameters xᵏ update
        - γₖ (gamma): momentum coefficient
        - ρₖ: learning rate
        - ∇f(xᵏ): gradient of the objective function at xᵏ

    Parameters
    ----------
    gamma : np.float64, optional
        Momentum coefficient controlling the contribution of previous updates.
        Default is 0.9.

    Attributes
    ----------
    momentum_term : ndarray
        Stores the momentum information from previous iterations.
    gamma : np.float64
        The momentum coefficient used in optimization.

    Notes
    -----
    The algorithm consists of two main steps:
        1. A descent step from point yᵏ to xᵏ along the gradient ∇f(yᵏ)
        2. A momentum step from xᵏ in the direction (xᵏ - xᵏ⁻¹)
    """

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
            self.momentum_term = np.zeros(shape=x.size)

        self.momentum_term = self.gamma * self.momentum_term + learning_rate * grad_fn(
            x - self.gamma * self.momentum_term
        )

        return self.momentum_term

    def reset(self) -> None:
        self.momentum_term = None


class AdagradOptimizer(BaseOptimizer):
    """AdaGrad (Adaptive Gradient) optimizer implementation. This optimizer
    addresses the problem of gradient optimization on sparse data by adapting
    the learning rate for each parameter. It normalizes the learning rate using
    accumulated gradients, giving more weight to infrequent parameters:

    xᵏ⁺¹ = xᵏ - ρ̃ₖ∇f(xᵏ), ρ̃ₖ = ρₖ / √(Gₖ + ε), Gₖ = Gₖ₋₁ + ∇f(xᵏ)²

    where:
        - k: iteration number
        - xᵏ: parameters at iteration k
        - ρ̃ₖ: normalized learning rate
        - ∇f(xᵏ): gradient of the objective function at xᵏ
        - Gₖ: accumulated sum of squared gradients from previous iterations
        - ε: a smoothing term to prevent zero in a normalizing denominator

    Parameters
    ----------
    var_eps : float64, optional
        Small constant for numerical stability (default: 1e-8)

    Attributes
    ----------
    grad_term : ndarray or None
        Accumulated sum of squared gradients
    var_eps : float64
        Numerical stability constant
    """

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
            self.grad_term = np.zeros(shape=x.size)

        grad_x = grad_fn(x)

        self.grad_term += grad_x**2

        return (learning_rate / np.sqrt(self.grad_term + self.var_eps)) * grad_x

    def reset(self) -> None:
        self.grad_term = None


class RMSPropOptimizer(BaseOptimizer):
    """RMSProp (Root Mean Squared Propagation) optimizer implementation. This optimizer
    solves the disadvantage of uncontrolled reduction of the learning rate ρₖ by calculating
    an expected value 𝔼[Gₖ] using moving average with a coefficient 0 < βₖ < 1:

    xᵏ⁺¹ = xᵏ - ρ̃ₖ∇f(xᵏ), ρ̃ₖ = ρₖ / √(Ḡₖ + ε), Ḡₖ = (1 - βₖ)Ḡₖ₋₁ + βₖ∇f(xᵏ)²

    where:
        - k: iteration number
        - xᵏ: parameters at iteration k
        - ρ̃ₖ: normalized learning rate
        - ∇f(xᵏ): gradient of the objective function at xᵏ
        - Ḡₖ: averaged sum of squared gradients from previous iterations
        - ε: a smoothing term to prevent zero in a normalizing denominator

    Parameters
    ----------
    beta : float64, optional
        Decay factor for the moving average of squared gradients (default: 0.9)
    var_eps : float64, optional
        Small constant for numerical stability (default: 1e-8)

    Attributes
    ----------
    grad_term : ndarray or None
        Running average of squared gradients
    beta : float64
        Decay factor for the moving average
    var_eps : float64
        Numerical stability constant

    Notes
    -----
    The algorithm automatically adjusts the learning rate for each parameter
    based on the historical magnitude of its gradients, helping with training
    stability.
    """

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
            self.grad_term = np.zeros(shape=x.size)

        grad_x = grad_fn(x)

        self.grad_term = self.beta * self.grad_term + (1 - self.beta) * grad_x**2

        return (learning_rate / np.sqrt(self.grad_term + self.var_eps)) * grad_x

    def reset(self) -> None:
        self.grad_term = None


class AdamOptimizer(BaseOptimizer):
    """Adam (Adaptive Moment Estimation) optimizer implementation. The optimizer
    combines ideas from RMSProp and momentum optimization, maintaining both
    first-order (mean) and second-order (variance) moments of the gradients.
    The update rules are:

    mₖ = β₁mₖ₋₁ + (1 - β₁)∇f(xᵏ)
    vₖ = β₂vₖ₋₁ + (1 - β₂)∇f(xᵏ)²
    xᵏ⁺¹ = xᵏ - ρ̃ₖ mₖ, ρ̃ₖ = ρₖ / √(vₖ + ε), k=1,...,K

    where:
        - k: iteration number
        - xᵏ: parameters at iteration k
        - ρ̃ₖ: normalized learning rate
        - ∇f(xᵏ): gradient of the objective function at xᵏ
        - mₖ: an expected value (the first momentum) of gradient values
        - vₖ: an unbiased variance (the second momentum) of gradient values
        - ε: a smoothing term to prevent zero in a normalizing denominator

    Parameters
    ----------
    betas : tuple of float, optional
        Coefficients used for computing moving averages of gradient and its square,
        default is (0.9, 0.999)
    var_eps : float, optional
        Small constant for numerical stability, default is 1e-8

    Attributes
    ----------
    momentum_term : ndarray
        Expected value (the first momentum) of gradient values
    variance_term : ndarray
        Unbiased variance (the second momentum) of gradient values
    betas : tuple of float
        Averaging coefficients for moment estimation
    var_eps : float
        Smoothing term to prevent zero in a normalizing denominator
    """

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
            self.momentum_term = np.zeros(shape=x.size)
            self.variance_term = np.zeros(shape=x.size)

        grad_x = grad_fn(x)

        self.momentum_term = beta1 * self.momentum_term + (1 - beta1) * grad_x
        self.variance_term = beta2 * self.variance_term + (1 - beta2) * grad_x**2

        return (
            learning_rate / np.sqrt(self.variance_term + self.var_eps)
        ) * self.momentum_term

    def reset(self) -> None:
        self.momentum_term = None
        self.variance_term = None
