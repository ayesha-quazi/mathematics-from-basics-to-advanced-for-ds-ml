# Linear Algebra for ML

## Overview
Linear algebra is the foundation of machine learning. It provides the mathematical framework for representing and manipulating data, models, and computations efficiently.

## Topics Covered

### 1. Vectors
- Vector representation
- Vector operations (addition, scalar multiplication)
- Dot product (inner product)
- Cross product
- Vector norms (L1, L2, infinity norm)
- Unit vectors and normalization

### 2. Matrices
- Matrix representation
- Matrix operations (addition, multiplication)
- Transpose
- Identity matrix
- Diagonal matrices
- Special matrices (symmetric, orthogonal)

### 3. Matrix Operations
- Matrix-vector multiplication
- Matrix-matrix multiplication
- Element-wise operations (Hadamard product)
- Kronecker product
- Broadcasting

### 4. Determinants
- Definition and properties
- Computing determinants
- Geometric interpretation
- Invertibility criterion

### 5. Matrix Inverse
- Definition of inverse
- Computing inverse (Gauss-Jordan elimination)
- Properties of inverse
- Pseudo-inverse (Moore-Penrose)

### 6. Solving Linear Systems
- Systems of linear equations: Ax = b
- Gaussian elimination
- LU decomposition
- Applications in ML (normal equations)

### 7. Matrix Decompositions
- LU decomposition
- QR decomposition
- Cholesky decomposition
- Applications in computation

## Applications in Machine Learning

### Data Representation
```python
# Data matrix X: (n_samples, n_features)
# Each row is a data point
# Each column is a feature
```

### Linear Regression
```python
# Normal equation: θ = (XᵀX)⁻¹Xᵀy
# Prediction: ŷ = Xθ
```

### Neural Networks
```python
# Layer computation: h = σ(Wx + b)
# W: weight matrix
# x: input vector
# b: bias vector
# σ: activation function
```

### Dimensionality Reduction
- PCA uses matrix decomposition
- Feature transformations

### Similarity and Distance
- Cosine similarity: (x·y)/(||x||·||y||)
- Euclidean distance: ||x - y||₂

## Files in this Directory

- `notes.md` - Detailed theory and formulas
- `vector_operations.py` - Vector operations with NumPy
- `matrix_operations.py` - Matrix operations and decompositions
- `ml_applications.py` - ML examples (linear regression, neural networks)
- `visualizations.ipynb` - Visual demonstrations

## Key Libraries
- NumPy: Fundamental array and matrix operations
- SciPy: Advanced linear algebra functions
- PyTorch/TensorFlow: Automatic differentiation with tensors
