"""
Algebra Basics - Python Examples
Demonstrates fundamental algebraic operations and concepts using Python
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def solve_linear_equation(a: float, b: float) -> float:
    """
    Solve linear equation ax + b = 0
    
    Args:
        a: coefficient of x
        b: constant term
        
    Returns:
        Solution x
        
    Example:
        >>> solve_linear_equation(2, -6)  # 2x - 6 = 0
        3.0
    """
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for a linear equation")
    return -b / a


def solve_quadratic(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """
    Solve quadratic equation ax² + bx + c = 0 using quadratic formula
    
    Args:
        a, b, c: coefficients
        
    Returns:
        Tuple of two solutions (may be complex)
        
    Example:
        >>> solve_quadratic(1, -5, 6)  # x² - 5x + 6 = 0
        (3.0, 2.0)
    """
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for a quadratic equation")
    
    discriminant = b**2 - 4*a*c
    sqrt_discriminant = np.sqrt(discriminant + 0j)  # Handle negative discriminant
    
    x1 = (-b + sqrt_discriminant) / (2*a)
    x2 = (-b - sqrt_discriminant) / (2*a)
    
    # Return real numbers if discriminant is non-negative
    if discriminant >= 0:
        return (float(x1.real), float(x2.real))
    return (x1, x2)


def evaluate_polynomial(coefficients: List[float], x: float) -> float:
    """
    Evaluate polynomial at given x value
    
    Args:
        coefficients: list of coefficients [a_n, a_{n-1}, ..., a_1, a_0]
                     for polynomial a_n*x^n + ... + a_1*x + a_0
        x: value at which to evaluate
        
    Returns:
        Polynomial value at x
        
    Example:
        >>> evaluate_polynomial([1, -2, 1], 3)  # x² - 2x + 1 at x=3
        4.0
    """
    return sum(coef * x**i for i, coef in enumerate(reversed(coefficients)))


def logarithm_properties_demo():
    """
    Demonstrate logarithm properties with examples
    """
    print("Logarithm Properties Demonstration")
    print("=" * 50)
    
    x, y = 8, 2
    
    # Product rule: log(xy) = log(x) + log(y)
    print(f"\n1. Product Rule: log({x}×{y}) = log({x}) + log({y})")
    print(f"   Left side:  log({x*y}) = {np.log(x*y):.4f}")
    print(f"   Right side: log({x}) + log({y}) = {np.log(x) + np.log(y):.4f}")
    
    # Quotient rule: log(x/y) = log(x) - log(y)
    print(f"\n2. Quotient Rule: log({x}/{y}) = log({x}) - log({y})")
    print(f"   Left side:  log({x/y}) = {np.log(x/y):.4f}")
    print(f"   Right side: log({x}) - log({y}) = {np.log(x) - np.log(y):.4f}")
    
    # Power rule: log(x^n) = n·log(x)
    n = 3
    print(f"\n3. Power Rule: log({x}^{n}) = {n}×log({x})")
    print(f"   Left side:  log({x**n}) = {np.log(x**n):.4f}")
    print(f"   Right side: {n}×log({x}) = {n * np.log(x):.4f}")
    
    # Change of base
    print(f"\n4. Change of Base: log₁₀({x}) = ln({x}) / ln(10)")
    print(f"   log₁₀({x}) = {np.log10(x):.4f}")
    print(f"   ln({x})/ln(10) = {np.log(x) / np.log(10):.4f}")


def plot_quadratic(a: float, b: float, c: float):
    """
    Plot quadratic function f(x) = ax² + bx + c
    """
    x = np.linspace(-10, 10, 400)
    y = a * x**2 + b * x + c
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {a}x² + {b}x + {c}')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Quadratic Function')
    plt.legend()
    
    # Mark roots if real
    try:
        roots = solve_quadratic(a, b, c)
        if all(isinstance(r, float) for r in roots):
            for root in roots:
                plt.plot(root, 0, 'ro', markersize=10, label=f'Root: x={root:.2f}')
    except:
        pass
    
    plt.legend()
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    print("Algebra Basics Examples\n")
    
    # Linear equation example
    print("1. Linear Equation: 3x - 9 = 0")
    solution = solve_linear_equation(3, -9)
    print(f"   Solution: x = {solution}\n")
    
    # Quadratic equation example
    print("2. Quadratic Equation: x² - 5x + 6 = 0")
    roots = solve_quadratic(1, -5, 6)
    print(f"   Roots: x₁ = {roots[0]}, x₂ = {roots[1]}\n")
    
    # Polynomial evaluation
    print("3. Polynomial: f(x) = 2x³ - 3x² + x - 5")
    coeffs = [2, -3, 1, -5]
    x_val = 2
    result = evaluate_polynomial(coeffs, x_val)
    print(f"   f({x_val}) = {result}\n")
    
    # Logarithm properties
    logarithm_properties_demo()
    
    print("\n" + "="*50)
    print("Examples completed successfully!")
