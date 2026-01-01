# Gradient Descent & Optimization

## Overview
Gradient descent and related optimization algorithms are the workhorses of machine learning. They enable us to train models by iteratively minimizing loss functions.

## Topics Covered

### 1. Gradient Descent Algorithm
- Basic concept: move in direction of steepest descent
- Update rule: θ = θ - α·∇L(θ)
- Learning rate (α)
- Convergence criteria
- Local vs global minima

### 2. Variants of Gradient Descent

#### Batch Gradient Descent
```python
# Uses entire dataset
# θ = θ - α·∇L(θ)
# Stable but slow for large datasets
```

#### Stochastic Gradient Descent (SGD)
```python
# Uses one sample at a time
# θ = θ - α·∇L(θ; xᵢ, yᵢ)
# Faster but noisy
```

#### Mini-batch Gradient Descent
```python
# Uses small batches
# θ = θ - α·∇L(θ; batch)
# Good trade-off between speed and stability
```

### 3. Learning Rate Strategies
- Fixed learning rate
- Learning rate decay
- Step decay
- Exponential decay
- Adaptive learning rates

### 4. Momentum-Based Methods

#### Momentum
```python
# v = β·v + ∇L(θ)
# θ = θ - α·v
# Accelerates convergence
```

#### Nesterov Accelerated Gradient (NAG)
```python
# Look-ahead gradient
# Better anticipation of changes
```

### 5. Adaptive Learning Rate Methods

#### AdaGrad
```python
# Adapts learning rate per parameter
# Good for sparse data
```

#### RMSprop
```python
# Uses moving average of squared gradients
# Addresses AdaGrad's diminishing learning rate
```

#### Adam (Adaptive Moment Estimation)
```python
# Combines momentum and RMSprop
# Most popular optimizer
# m = β₁·m + (1-β₁)·∇L(θ)
# v = β₂·v + (1-β₂)·(∇L(θ))²
```

#### AdamW
```python
# Adam with weight decay
# Better regularization
```

### 6. Second-Order Methods
- Newton's method
- Quasi-Newton methods (BFGS, L-BFGS)
- Conjugate gradient
- Trade-off: speed vs computation cost

### 7. Convex vs Non-Convex Optimization

#### Convex Optimization
- Single global minimum
- Guaranteed convergence
- Examples: linear regression, logistic regression

#### Non-Convex Optimization
- Multiple local minima
- Saddle points
- Examples: neural networks
- Challenges and techniques

### 8. Regularization in Optimization
- L1 regularization (Lasso): promotes sparsity
- L2 regularization (Ridge): prevents overfitting
- Elastic Net: combines L1 and L2
- Dropout in neural networks

### 9. Gradient Clipping
- Prevents exploding gradients
- Clip by value or by norm
- Important for RNNs and deep networks

### 10. Optimization Challenges
- Vanishing gradients
- Exploding gradients
- Saddle points
- Plateaus
- Local minima

## Practical Considerations

### Hyperparameters
- Learning rate: most critical parameter
- Batch size: affects convergence and memory
- Momentum parameters (β₁, β₂)
- Weight decay coefficient

### Monitoring Convergence
- Loss curves
- Gradient norms
- Learning rate schedules
- Early stopping

### Initialization
- Xavier/Glorot initialization
- He initialization
- Impact on convergence

## Applications in Machine Learning

### Training Neural Networks
```python
# Forward pass → Compute loss
# Backward pass → Compute gradients
# Update weights using optimizer
```

### Linear Models
```python
# Logistic regression
# Support Vector Machines
```

### Deep Learning
```python
# CNNs, RNNs, Transformers
# Different architectures, same optimization principles
```

### Online Learning
```python
# Update model as new data arrives
# SGD is natural choice
```

## Files in this Directory

- `notes.md` - Detailed theory and mathematical derivations
- `gradient_descent.py` - Implementation from scratch
- `optimizers.py` - Various optimizer implementations
- `comparison.ipynb` - Comparing different optimizers
- `visualization.ipynb` - Visualizing optimization paths
- `ml_examples.py` - Training models with different optimizers
- `hyperparameter_tuning.ipynb` - Learning rate and batch size effects

## Key Libraries
- NumPy: For basic implementations
- PyTorch: torch.optim (SGD, Adam, etc.)
- TensorFlow/Keras: tf.keras.optimizers
- JAX: jax.example_libraries.optimizers

## Key Takeaways
- Adam is a good default choice for most problems
- Learning rate is the most important hyperparameter
- Use learning rate schedules for better convergence
- Monitor gradients to detect training issues
- Different problems may benefit from different optimizers
