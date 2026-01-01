# Differentiation

## Overview
Differentiation is the process of finding rates of change. In ML, it's essential for computing gradients and optimizing models.

## Topics Covered

### 1. Limits and Continuity
- Definition of limits
- Properties of limits
- Continuity and discontinuity
- Epsilon-delta definition

### 2. Derivatives Basics
- Definition of derivative
- Derivative as rate of change
- Derivative notation (dy/dx, f'(x), ∂f/∂x)

### 3. Differentiation Rules
- Power rule: d/dx(xⁿ) = nxⁿ⁻¹
- Constant rule
- Sum and difference rules
- Product rule: (fg)' = f'g + fg'
- Quotient rule: (f/g)' = (f'g - fg')/g²
- Chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x)

### 4. Common Derivatives
- Exponential: d/dx(eˣ) = eˣ
- Logarithmic: d/dx(ln x) = 1/x
- Trigonometric functions
- Inverse trigonometric functions

### 5. Partial Derivatives
- Functions of multiple variables
- Partial derivative notation: ∂f/∂x
- Mixed partial derivatives
- Gradient vector: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

### 6. Higher-Order Derivatives
- Second derivatives
- Hessian matrix
- Concavity and inflection points

## Applications in Machine Learning

### Loss Function Gradients
```python
# Example: Mean Squared Error
# Loss = (1/n) * Σ(y - ŷ)²
# ∂Loss/∂w = -(2/n) * Σ(y - ŷ) * x
```

### Activation Function Derivatives
- Sigmoid: σ'(x) = σ(x)(1 - σ(x))
- ReLU: f'(x) = 1 if x > 0, else 0
- Tanh: tanh'(x) = 1 - tanh²(x)

### Backpropagation
Computing gradients layer by layer using the chain rule.

## Files in this Directory

- `notes.md` - Detailed theory and formulas
- `derivative_rules.py` - Implementation of derivative rules
- `gradient_examples.py` - ML gradient examples
- `visualization.ipynb` - Visualizing derivatives
