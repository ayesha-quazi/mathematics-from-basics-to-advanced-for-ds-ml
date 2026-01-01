"""
Linear Algebra for ML - Python Examples
Demonstrates core linear algebra operations used in machine learning
"""

import numpy as np
from typing import Tuple
from scipy.linalg import lu


def vector_operations_demo():
    """Demonstrate basic vector operations"""
    print("Vector Operations")
    print("=" * 50)
    
    # Create vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}\n")
    
    # Vector addition
    print(f"Addition: v1 + v2 = {v1 + v2}")
    
    # Scalar multiplication
    print(f"Scalar multiplication: 3 * v1 = {3 * v1}")
    
    # Dot product
    dot_product = np.dot(v1, v2)
    print(f"Dot product: v1 · v2 = {dot_product}")
    
    # Vector norms
    l1_norm = np.linalg.norm(v1, ord=1)
    l2_norm = np.linalg.norm(v1, ord=2)
    print(f"L1 norm: ||v1||₁ = {l1_norm}")
    print(f"L2 norm: ||v1||₂ = {l2_norm:.4f}")
    
    # Unit vector
    unit_vector = v1 / l2_norm
    print(f"Unit vector: v1/||v1|| = {unit_vector}")
    print()


def matrix_operations_demo():
    """Demonstrate basic matrix operations"""
    print("Matrix Operations")
    print("=" * 50)
    
    # Create matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print()
    
    # Matrix addition
    print("A + B =")
    print(A + B)
    print()
    
    # Matrix multiplication
    print("A @ B (matrix multiplication) =")
    print(A @ B)
    print()
    
    # Element-wise multiplication
    print("A * B (element-wise, Hadamard product) =")
    print(A * B)
    print()
    
    # Transpose
    print("Aᵀ (transpose) =")
    print(A.T)
    print()
    
    # Determinant
    det_A = np.linalg.det(A)
    print(f"det(A) = {det_A}")
    print()
    
    # Inverse
    A_inv = np.linalg.inv(A)
    print("A⁻¹ (inverse) =")
    print(A_inv)
    print("\nVerify A @ A⁻¹ = I:")
    print(A @ A_inv)
    print()


def solve_linear_system():
    """Solve system of linear equations Ax = b"""
    print("Solving Linear System Ax = b")
    print("=" * 50)
    
    # System: 2x + 3y = 8
    #         5x + 4y = 13
    A = np.array([[2, 3], [5, 4]])
    b = np.array([8, 13])
    
    print("System:")
    print("2x + 3y = 8")
    print("5x + 4y = 13")
    print()
    
    # Solve using NumPy
    x = np.linalg.solve(A, b)
    print(f"Solution: x = {x[0]:.2f}, y = {x[1]:.2f}")
    
    # Verify
    print(f"Verification: A @ x = {A @ x}")
    print(f"Expected: b = {b}")
    print()


def matrix_decompositions():
    """Demonstrate matrix decompositions"""
    print("Matrix Decompositions")
    print("=" * 50)
    
    A = np.array([[4, 2], [1, 3]])
    print("Matrix A:")
    print(A)
    print()
    
    # LU Decomposition
    P, L, U = lu(A)
    print("LU Decomposition: PA = LU")
    print("L (lower triangular):")
    print(L)
    print("\nU (upper triangular):")
    print(U)
    print()
    
    # QR Decomposition
    Q, R = np.linalg.qr(A)
    print("QR Decomposition: A = QR")
    print("Q (orthogonal):")
    print(Q)
    print("\nR (upper triangular):")
    print(R)
    print()
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("Eigendecomposition:")
    print(f"Eigenvalues: {eigenvalues}")
    print("Eigenvectors:")
    print(eigenvectors)
    print()
    
    # SVD
    U, S, Vt = np.linalg.svd(A)
    print("Singular Value Decomposition: A = UΣVᵀ")
    print("U:")
    print(U)
    print(f"\nΣ (singular values): {S}")
    print("\nVᵀ:")
    print(Vt)
    print()


class LinearRegressionNormalEquation:
    """
    Linear Regression using Normal Equation
    θ = (XᵀX)⁻¹Xᵀy
    """
    
    def __init__(self):
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit linear regression model
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Add bias term
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Normal equation: θ = (XᵀX)⁻¹Xᵀy
        self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return X_with_bias @ self.weights


def linear_regression_example():
    """Example of linear regression using normal equation"""
    print("Linear Regression using Normal Equation")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Fit model
    model = LinearRegressionNormalEquation()
    model.fit(X, y)
    
    print(f"True model: y = 4 + 3x + noise")
    print(f"Learned weights: {model.weights}")
    print(f"Learned model: y = {model.weights[0]:.2f} + {model.weights[1]:.2f}x")
    
    # Make predictions
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")
    print()


def cosine_similarity_example():
    """Demonstrate cosine similarity for ML applications"""
    print("Cosine Similarity")
    print("=" * 50)
    
    # Two document vectors (word frequencies)
    doc1 = np.array([1, 2, 1, 0, 3])  # "machine learning is great machine"
    doc2 = np.array([1, 1, 2, 1, 2])  # "machine learning data science learning"
    doc3 = np.array([0, 0, 0, 5, 0])  # "science science science science science"
    
    def cosine_similarity(v1, v2):
        """Compute cosine similarity between two vectors"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    sim_12 = cosine_similarity(doc1, doc2)
    sim_13 = cosine_similarity(doc1, doc3)
    sim_23 = cosine_similarity(doc2, doc3)
    
    print(f"Similarity between doc1 and doc2: {sim_12:.4f}")
    print(f"Similarity between doc1 and doc3: {sim_13:.4f}")
    print(f"Similarity between doc2 and doc3: {sim_23:.4f}")
    print("\nDoc1 and Doc2 are most similar (both about ML)")
    print()


if __name__ == "__main__":
    print("Linear Algebra for ML - Examples\n")
    
    # Run all demonstrations
    vector_operations_demo()
    matrix_operations_demo()
    solve_linear_system()
    matrix_decompositions()
    linear_regression_example()
    cosine_similarity_example()
    
    print("=" * 50)
    print("All examples completed successfully!")
