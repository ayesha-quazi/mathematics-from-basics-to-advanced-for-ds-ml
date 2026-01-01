# Optimization

## Overview
Optimization is about finding the best solution from all feasible solutions. In ML, we optimize loss functions to train models.

## Topics Covered

### 1. Critical Points
- Finding critical points: f'(x) = 0
- Stationary points
- Classification using derivatives

### 2. Maxima and Minima
- Local vs global extrema
- First derivative test
- Second derivative test
- Concavity and inflection points

### 3. Multivariable Optimization
- Partial derivatives
- Gradient: ∇f
- Critical points for functions of multiple variables
- Saddle points

### 4. Constrained Optimization
- Lagrange multipliers
- KKT conditions
- Penalty methods

### 5. Convexity
- Convex functions
- Convex sets
- Properties of convex optimization
- Global optimality conditions

## Optimization Methods

### Analytical Methods
- Setting derivative to zero
- Solving system of equations

### Numerical Methods
- Newton's method
- Quasi-Newton methods (BFGS)
- Conjugate gradient

## Applications in Machine Learning

### Loss Function Minimization
```python
# Minimize: L(θ) = (1/n)Σ loss(yᵢ, f(xᵢ; θ))
# Find: θ* = argmin L(θ)
```

### Regularization
```python
# L2 regularization: L(θ) + λ||θ||²
# L1 regularization: L(θ) + λ||θ||₁
```

### Constrained ML Problems
- Support Vector Machines (SVM)
- Portfolio optimization
- Resource allocation

### Hyperparameter Tuning
Finding optimal hyperparameters using:
- Grid search
- Random search
- Bayesian optimization

## Files in this Directory

- `notes.md` - Detailed theory and formulas
- `optimization_methods.py` - Implementation of methods
- `ml_optimization.py` - ML-specific examples
- `visualizations.ipynb` - Visual demonstrations
