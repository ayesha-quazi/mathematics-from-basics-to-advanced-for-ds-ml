# Numerical Methods & Transformations

## Overview
Numerical methods provide computational techniques for solving mathematical problems that may not have analytical solutions. Transformations help analyze signals and data in different domains.

## Topics Covered

### 1. Numerical Differentiation
- Finite difference methods
- Forward difference: f'(x) ≈ [f(x+h) - f(x)]/h
- Backward difference: f'(x) ≈ [f(x) - f(x-h)]/h
- Central difference: f'(x) ≈ [f(x+h) - f(x-h)]/(2h)
- Error analysis
- Applications in computing gradients

### 2. Numerical Integration
- Riemann sums (left, right, midpoint)
- Trapezoidal rule
- Simpson's rule
- Adaptive quadrature
- Monte Carlo integration
- Applications in probability

### 3. Root Finding Methods

#### Bisection Method
```python
# Bracketing method
# Guaranteed convergence (slow)
```

#### Newton-Raphson Method
```python
# x_{n+1} = x_n - f(x_n)/f'(x_n)
# Fast convergence (requires derivative)
```

#### Secant Method
```python
# Approximates derivative
# No derivative needed
```

#### Fixed-Point Iteration
```python
# x = g(x)
# Iterative solution
```

### 4. Solving Linear Systems Numerically
- Direct methods (Gaussian elimination, LU)
- Iterative methods (Jacobi, Gauss-Seidel)
- Conjugate gradient method
- Preconditioners
- Sparse matrix techniques

### 5. Interpolation
- Lagrange interpolation
- Newton's divided differences
- Spline interpolation (cubic splines)
- Applications in data analysis

### 6. Approximation Theory
- Polynomial approximation
- Least squares approximation
- Chebyshev polynomials
- Function fitting

### 7. Fourier Transform

#### Discrete Fourier Transform (DFT)
```python
# X[k] = Σ x[n]·e^(-2πikn/N)
# Converts time domain → frequency domain
```

#### Fast Fourier Transform (FFT)
```python
# Efficient O(N log N) algorithm
# Implementation in NumPy/SciPy
```

#### Applications
- Signal processing
- Audio analysis
- Image processing
- Feature extraction in ML

### 8. Laplace Transform
- Definition: L{f(t)} = ∫₀^∞ f(t)e^(-st)dt
- Properties and theorems
- Solving differential equations
- Transfer functions

### 9. Z-Transform
- Definition for discrete signals
- Region of convergence
- Applications in digital signal processing
- Relationship to Fourier transform

### 10. Wavelet Transforms
- Continuous wavelet transform
- Discrete wavelet transform
- Multi-resolution analysis
- Applications in time-frequency analysis

### 11. Other Important Transforms
- Discrete Cosine Transform (DCT)
- Hilbert Transform
- Radon Transform (CT scans)

### 12. Numerical Stability
- Condition number
- Round-off errors
- Truncation errors
- Stability analysis

## Applications in Machine Learning and Data Science

### Feature Engineering
```python
# FFT for time series features
# Spectral features for audio
# Wavelet features for signals
```

### Signal Processing
```python
# Filtering noise
# Extracting patterns
# Compression
```

### Image Processing
```python
# Image compression (DCT in JPEG)
# Edge detection
# Frequency analysis
```

### Time Series Analysis
```python
# Spectral analysis
# Seasonality detection
# Trend analysis
```

### Numerical Optimization
```python
# Computing gradients numerically
# Solving optimization problems
# Root finding in algorithms
```

### Scientific Computing
```python
# Simulations
# Differential equations in physics-informed ML
# Numerical PDEs
```

## Files in this Directory

- `notes.md` - Detailed theory and formulas
- `numerical_differentiation.py` - Finite differences
- `numerical_integration.py` - Integration methods
- `root_finding.py` - Root finding algorithms
- `fourier_transform.py` - FFT examples
- `wavelet_transform.py` - Wavelet analysis
- `interpolation.py` - Interpolation methods
- `ml_applications.ipynb` - Feature engineering with transforms
- `signal_processing.ipynb` - Audio/signal analysis
- `visualizations.ipynb` - Visualizing transforms

## Key Libraries
- NumPy: Basic numerical operations, FFT
- SciPy: scipy.integrate, scipy.optimize, scipy.signal
- PyWavelets: Wavelet transforms
- Scikit-learn: Numerical methods in ML algorithms

## Key Concepts for ML
- Fourier transform is crucial for signal and time series analysis
- Numerical methods enable gradient computation in deep learning
- Transforms provide alternative representations of data
- Understanding numerical stability is important for robust implementations
- Interpolation and approximation are used in various ML algorithms
