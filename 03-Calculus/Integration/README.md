# Integration

## Overview
Integration is the inverse process of differentiation. In ML, it's used in probability theory, computing areas under curves, and understanding cumulative distributions.

## Topics Covered

### 1. Antiderivatives
- Indefinite integrals: ∫f(x)dx
- Constant of integration
- Basic integration rules

### 2. Definite Integrals
- Definition: ∫ᵃᵇ f(x)dx
- Fundamental Theorem of Calculus
- Properties of definite integrals

### 3. Integration Techniques
- Substitution method (u-substitution)
- Integration by parts: ∫udv = uv - ∫vdu
- Partial fractions
- Trigonometric substitution

### 4. Common Integrals
- Power rule: ∫xⁿdx = xⁿ⁺¹/(n+1) + C
- Exponential: ∫eˣdx = eˣ + C
- Logarithmic: ∫(1/x)dx = ln|x| + C
- Trigonometric integrals

### 5. Numerical Integration
- Riemann sums
- Trapezoidal rule
- Simpson's rule
- Monte Carlo integration

### 6. Multiple Integrals
- Double integrals
- Triple integrals
- Applications in probability

## Applications in Machine Learning

### Probability Distributions
```python
# Probability density functions (PDF)
# ∫₋∞^∞ f(x)dx = 1 (normalization)

# Cumulative distribution function (CDF)
# F(x) = ∫₋∞^x f(t)dt
```

### Expected Value
```python
# E[X] = ∫₋∞^∞ x·f(x)dx
```

### Area Under ROC Curve
Computing model performance metrics.

### Computing Marginal Distributions
Integrating joint probability distributions.

## Files in this Directory

- `notes.md` - Detailed theory and formulas
- `integration_techniques.py` - Implementation examples
- `numerical_integration.py` - Numerical methods
- `probability_applications.ipynb` - ML applications
