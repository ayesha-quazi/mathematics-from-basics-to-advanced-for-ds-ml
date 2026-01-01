# Vector Spaces, Eigenvalues, and SVD

## Overview
Advanced linear algebra concepts that are crucial for dimensionality reduction, feature extraction, and understanding the structure of data in machine learning.

## Topics Covered

### 1. Vector Spaces
- Definition of vector space
- Subspaces
- Span of vectors
- Linear combinations
- Column space and row space
- Null space (kernel)

### 2. Linear Independence
- Definition
- Testing for linear independence
- Basis of a vector space
- Coordinates with respect to a basis

### 3. Basis and Dimension
- Standard basis
- Orthogonal basis
- Orthonormal basis
- Dimension of a vector space
- Rank of a matrix

### 4. Inner Products and Orthogonality
- Inner product spaces
- Orthogonal vectors
- Orthogonal projection
- Gram-Schmidt process
- QR decomposition

### 5. Eigenvalues and Eigenvectors
- Definition: Av = λv
- Characteristic equation: det(A - λI) = 0
- Computing eigenvalues and eigenvectors
- Eigendecomposition: A = QΛQ⁻¹
- Properties of eigenvalues
- Trace and determinant

### 6. Diagonalization
- Diagonalizable matrices
- Similar matrices
- Powers of matrices
- Applications in solving systems

### 7. Singular Value Decomposition (SVD)
- Definition: A = UΣVᵀ
- Computing SVD
- Relationship to eigendecomposition
- Reduced SVD vs Full SVD
- Properties and applications

### 8. Principal Component Analysis (PCA)
- Motivation: dimensionality reduction
- Mathematical formulation
- Computing principal components
- Variance explained
- Reconstruction error
- PCA using eigendecomposition
- PCA using SVD

### 9. Matrix Norms and Condition Numbers
- Frobenius norm
- Spectral norm
- Condition number
- Numerical stability

## Applications in Machine Learning

### Dimensionality Reduction
```python
# PCA for feature reduction
# X_reduced = X @ V_k  # Project onto top k components
```

### Recommendation Systems
```python
# Matrix factorization
# R ≈ UΣVᵀ (user-item ratings)
```

### Image Compression
```python
# Using SVD to compress images
# Keep only top k singular values
```

### Data Whitening
```python
# Decorrelate features
# Transform to unit variance
```

### Latent Semantic Analysis (LSA)
- Text mining and NLP
- Topic modeling

### Face Recognition (Eigenfaces)
- Representing faces in lower dimensions
- Using PCA for recognition

### Spectral Clustering
- Using eigenvectors of similarity matrices
- Graph-based clustering

## Files in this Directory

- `notes.md` - Detailed theory and formulas
- `vector_spaces.py` - Vector space operations
- `eigenvalues.py` - Eigenvalue computations
- `svd_examples.py` - SVD implementations
- `pca_implementation.py` - PCA from scratch and with sklearn
- `applications.ipynb` - Real-world ML applications
- `image_compression.ipynb` - SVD for image compression
- `visualizations.ipynb` - Visualizing concepts

## Key Concepts for ML
- PCA is one of the most important dimensionality reduction techniques
- SVD is used in collaborative filtering and recommender systems
- Eigendecomposition helps understand dynamics of systems
- Understanding data structure through subspaces

## Key Libraries
- NumPy: np.linalg for eigenvalues, SVD
- SciPy: scipy.linalg for advanced operations
- Scikit-learn: PCA, TruncatedSVD implementations
