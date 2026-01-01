# Probability & Statistics

## Overview
Probability and statistics form the theoretical foundation for understanding uncertainty, making predictions, and evaluating model performance in machine learning.

## Topics Covered

### 1. Probability Basics
- Sample space and events
- Probability axioms
- Conditional probability: P(A|B)
- Independence
- Bayes' theorem
- Law of total probability

### 2. Random Variables
- Discrete random variables
- Continuous random variables
- Probability mass function (PMF)
- Probability density function (PDF)
- Cumulative distribution function (CDF)

### 3. Probability Distributions

#### Discrete Distributions
- Bernoulli distribution
- Binomial distribution
- Poisson distribution
- Geometric distribution

#### Continuous Distributions
- Uniform distribution
- Normal (Gaussian) distribution
- Exponential distribution
- Beta distribution
- Gamma distribution

### 4. Expected Value and Variance
- Expected value: E[X] = Σ x·P(X=x)
- Variance: Var(X) = E[(X - μ)²]
- Standard deviation
- Covariance: Cov(X,Y)
- Correlation: ρ(X,Y)

### 5. Joint and Marginal Distributions
- Joint probability: P(X,Y)
- Marginal probability
- Conditional distributions
- Independence of random variables

### 6. Central Limit Theorem
- Statement and importance
- Sampling distributions
- Applications in inference

### 7. Descriptive Statistics
- Mean, median, mode
- Quartiles and percentiles
- Range, IQR
- Skewness and kurtosis

### 8. Inferential Statistics
- Point estimation
- Interval estimation (confidence intervals)
- Hypothesis testing (t-test, z-test, chi-square)
- p-values and significance levels
- Type I and Type II errors

### 9. Statistical Tests
- t-tests (one-sample, two-sample, paired)
- ANOVA
- Chi-square test
- Non-parametric tests (Mann-Whitney, Wilcoxon)

### 10. Regression Analysis
- Simple linear regression
- Multiple linear regression
- Correlation coefficient
- R² (coefficient of determination)
- Residual analysis

## Applications in Machine Learning

### Probabilistic Models
```python
# Naive Bayes: P(y|x) ∝ P(x|y)·P(y)
# Maximum Likelihood Estimation (MLE)
# Maximum A Posteriori (MAP)
```

### Bayesian Machine Learning
- Prior and posterior distributions
- Bayesian inference
- Bayesian networks

### Model Evaluation
```python
# Cross-validation
# Confidence intervals for metrics
# Statistical significance testing
```

### Uncertainty Quantification
- Prediction intervals
- Confidence in predictions
- Ensemble methods

### Feature Selection
- Correlation analysis
- Chi-square test for independence
- ANOVA for feature importance

## Files in this Directory

- `notes.md` - Detailed theory and formulas
- `distributions.py` - Implementation of distributions
- `statistical_tests.py` - Hypothesis testing examples
- `bayesian_methods.py` - Bayesian inference
- `ml_applications.ipynb` - ML-specific applications
- `visualizations.ipynb` - Distribution visualizations

## Key Libraries
- NumPy: Basic statistical operations
- SciPy.stats: Statistical distributions and tests
- Statsmodels: Advanced statistical modeling
- PyMC3/Stan: Bayesian inference
