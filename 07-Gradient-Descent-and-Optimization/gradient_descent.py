"""
Gradient Descent Implementation
Demonstrates various gradient descent algorithms from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


class GradientDescent:
    """Basic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.history = []
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step
        
        Args:
            params: Current parameters
            gradients: Gradients of loss with respect to parameters
            
        Returns:
            Updated parameters
        """
        updated_params = params - self.learning_rate * gradients
        return updated_params


class MomentumGD:
    """Gradient Descent with Momentum"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Perform one optimization step with momentum"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity + gradients
        updated_params = params - self.learning_rate * self.velocity
        return updated_params


class Adam:
    """Adam optimizer (Adaptive Moment Estimation)"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Perform one optimization step with Adam"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        updated_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_params


def optimize(
    loss_fn: Callable,
    gradient_fn: Callable,
    initial_params: np.ndarray,
    optimizer,
    num_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Optimize parameters using given optimizer
    
    Args:
        loss_fn: Function to compute loss
        gradient_fn: Function to compute gradients
        initial_params: Starting parameters
        optimizer: Optimizer instance
        num_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (optimized parameters, loss history)
    """
    params = initial_params.copy()
    loss_history = []
    
    for i in range(num_iterations):
        # Compute loss and gradients
        loss = loss_fn(params)
        gradients = gradient_fn(params)
        
        loss_history.append(loss)
        
        # Check convergence
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        # Update parameters
        params = optimizer.step(params, gradients)
    
    return params, loss_history


# Example: Optimize a simple quadratic function
def quadratic_loss(params: np.ndarray) -> float:
    """f(x, y) = x² + y²"""
    return np.sum(params ** 2)


def quadratic_gradient(params: np.ndarray) -> np.ndarray:
    """Gradient of f(x, y) = x² + y²"""
    return 2 * params


# Example: Linear regression using gradient descent
class LinearRegressionGD:
    """Linear Regression with Gradient Descent"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit linear regression model
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Predictions
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw = -(2/n_samples) * X.T.dot(y - y_pred)
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return X.dot(self.weights) + self.bias


def compare_optimizers():
    """Compare different optimizers on the same problem"""
    initial_params = np.array([10.0, 10.0])
    
    optimizers = {
        'GD': GradientDescent(learning_rate=0.1),
        'Momentum': MomentumGD(learning_rate=0.1, momentum=0.9),
        'Adam': Adam(learning_rate=0.5)
    }
    
    plt.figure(figsize=(12, 4))
    
    for idx, (name, optimizer) in enumerate(optimizers.items(), 1):
        params, history = optimize(
            quadratic_loss,
            quadratic_gradient,
            initial_params,
            optimizer,
            num_iterations=100
        )
        
        plt.subplot(1, 3, idx)
        plt.plot(history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{name} Optimizer')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    print("Gradient Descent Examples\n")
    
    # Example 1: Simple optimization
    print("1. Optimizing f(x, y) = x² + y²")
    initial = np.array([5.0, 5.0])
    gd = GradientDescent(learning_rate=0.1)
    result, history = optimize(quadratic_loss, quadratic_gradient, initial, gd, 100)
    print(f"   Initial: {initial}")
    print(f"   Optimized: {result}")
    print(f"   Final loss: {history[-1]:.6f}\n")
    
    # Example 2: Linear regression
    print("2. Linear Regression with Gradient Descent")
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1
    
    model = LinearRegressionGD(learning_rate=0.1, num_iterations=1000)
    model.fit(X, y)
    print(f"   True: y = 3x + 2")
    print(f"   Learned: y = {model.weights[0]:.2f}x + {model.bias:.2f}")
    print(f"   Final loss: {model.loss_history[-1]:.6f}\n")
    
    print("Examples completed successfully!")
