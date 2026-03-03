# Module 3.2: Linear Models

## **Why Linear Models Matter**

**Linear models are the foundation of machine learning.**

- **Simple but powerful**: Often work surprisingly well
- **Fast**: Train quickly even on large datasets
- **Interpretable**: Understand exactly what model does
- **Foundation**: Concepts used in neural networks, LLMs
- **Interview favorite**: Always asked in ML interviews

Master linear models = understand core ML principles.

---

## **CONCEPT 1: Linear Regression**

### **The Simplest ML Algorithm**

**Goal**: Find a line (or plane) that best fits the data.

**1D Example (one feature):**
```
Data points: (x, y)
Find line: y = mx + b

m = slope (weight)
b = intercept (bias)
```

**Visual:**
```
Price
  ^
  |     *
  |   *   *
  | *   /
  |   / *
  | /
  +--------> Size
  
  Line: price = m × size + b
```

---

### **Multiple Features (Multiple Linear Regression)**

```
y = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b

y = prediction
x₁, x₂, ... = features
w₁, w₂, ... = weights (coefficients)
b = bias (intercept)
```

**Example: House Price Prediction**
```
price = w₁×size + w₂×bedrooms + w₃×age + w₄×location_score + b

If: w₁=200, w₂=10000, w₃=-500, w₄=5000, b=50000

For house: size=2000, bedrooms=3, age=10, location=8
price = 200×2000 + 10000×3 + (-500)×10 + 5000×8 + 50000
      = 400000 + 30000 - 5000 + 40000 + 50000
      = $515,000
```

---

### **Matrix Form (NumPy Form)**

```
ŷ = Xw + b

X = feature matrix (n_samples × n_features)
w = weight vector (n_features × 1)
b = bias (scalar)
ŷ = predictions (n_samples × 1)
```

**Or combining bias into weights:**
```
ŷ = Xw

where X has extra column of 1s for bias
```

---

## **CONCEPT 2: Cost Function (Loss Function)**

### **How Do We Measure "Best Fit"?**

**Mean Squared Error (MSE):**
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

yᵢ = actual value
ŷᵢ = predicted value
```

**Why squared?**
- Penalizes large errors more
- Always positive
- Differentiable (can use calculus)
- Mathematically convenient

**Goal**: Find weights that minimize MSE!

---

### **Visualization**

```
For simple case: y = mx + b

Try different values of m and b
Calculate MSE for each
Find m and b that give lowest MSE

This is a "loss landscape"
```

---

## **CONCEPT 3: Gradient Descent**

### **How to Find Best Weights**

**Analogy**: You're in foggy mountains, trying to reach the valley.
- Can't see the whole landscape
- Can only feel the slope under your feet
- Walk downhill until you reach the bottom

**Algorithm:**
```
1. Start with random weights
2. Calculate gradient (slope of loss function)
3. Update weights: w = w - learning_rate × gradient
4. Repeat until convergence
```

---

### **Mathematics**

For MSE loss: `L = (1/n) × Σ(y - Xw)²`

Gradient with respect to weights:
```
∂L/∂w = -(2/n) × Xᵀ(y - Xw)

This tells us: which direction to move weights
```

**Update rule:**
```
w_new = w_old - α × ∂L/∂w

α = learning rate (step size)
```

---

### **Learning Rate**

**Critical hyperparameter!**

```
Too small:
- Training very slow
- Takes forever to converge

Good:
- Steady progress
- Converges smoothly

Too large:
- Unstable training
- Might diverge (get worse)
- Jumps around, never settles
```

---

### **Gradient Descent Variants**

**1. Batch Gradient Descent**
```
Use ALL data to calculate gradient
One update per pass through dataset

Pros: Accurate gradient
Cons: Slow for large datasets
```

**2. Stochastic Gradient Descent (SGD)**
```
Use ONE random sample to calculate gradient
Many updates per pass

Pros: Fast, can escape local minima
Cons: Noisy, unstable
```

**3. Mini-Batch Gradient Descent** (most common)
```
Use small batch (32, 64, 128 samples) to calculate gradient
Good balance

Pros: Fast + stable
Cons: Need to choose batch size
```

---

## **CONCEPT 4: Normal Equation (Analytical Solution)**

### **Alternative to Gradient Descent**

For linear regression, there's a direct formula!

```
w = (XᵀX)⁻¹Xᵀy

Where:
Xᵀ = transpose of X
⁻¹ = matrix inverse
```

**Pros:**
- One-step solution
- No hyperparameters (no learning rate)
- Exact answer

**Cons:**
- Computing (XᵀX)⁻¹ is slow for many features
- O(n³) complexity - not scalable
- Doesn't work for other models

**When to use:**
- Small datasets (< 10,000 features)
- Quick baseline
- Guaranteed optimal solution

---

## **CONCEPT 5: Logistic Regression**

### **Linear Models for Classification**

**Problem**: Linear regression can predict any value.
For classification, we need output between 0 and 1 (probability).

**Solution**: Add sigmoid function!

---

### **Sigmoid Function**

```
σ(z) = 1 / (1 + e⁻ᶻ)

Properties:
- Input: any real number
- Output: between 0 and 1
- Smooth, differentiable
```

**Shape:**
```
σ(z)
 1 |        ________
   |       /
0.5|      /
   |     /
 0 |____/
   |
   +-----|-----|----- z
        -5     5
```

**For z >> 0**: σ(z) ≈ 1
**For z << 0**: σ(z) ≈ 0
**For z = 0**: σ(z) = 0.5

---

### **Logistic Regression Model**

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
P(y=1|x) = σ(z) = 1 / (1 + e⁻ᶻ)

If P(y=1|x) > 0.5: predict class 1
If P(y=1|x) ≤ 0.5: predict class 0
```

**Example: Spam Detection**
```
Features: [num_exclamation, has_word_free, has_word_urgent]
Weights: [0.5, 1.2, 0.8]
Bias: -2.0

Email: [3, 1, 0]  (3 exclamations, has "free", no "urgent")

z = 0.5×3 + 1.2×1 + 0.8×0 + (-2.0)
  = 1.5 + 1.2 + 0 - 2.0
  = 0.7

P(spam) = 1 / (1 + e⁻⁰·⁷) = 0.668 or 66.8%

Since 0.668 > 0.5 → Predict SPAM
```

---

### **Cross-Entropy Loss**

Can't use MSE for classification! Use cross-entropy instead.

**Binary Cross-Entropy:**
```
Loss = -[y×log(ŷ) + (1-y)×log(1-ŷ)]

For one sample:
If y=1: Loss = -log(ŷ)
If y=0: Loss = -log(1-ŷ)

Average over all samples:
L = -(1/n) × Σ[yᵢ×log(ŷᵢ) + (1-yᵢ)×log(1-ŷᵢ)]
```

**Why?**
- Penalizes confident wrong predictions heavily
- Gradient descent works well
- Derived from maximum likelihood estimation

**Intuition:**
```
If actual=1, predicted=0.9 → Loss small (good)
If actual=1, predicted=0.1 → Loss large (bad)
If actual=0, predicted=0.1 → Loss small (good)
If actual=0, predicted=0.9 → Loss large (bad)
```

---

## **CONCEPT 6: Regularization**

### **Preventing Overfitting**

**Problem**: With many features, model can overfit.

**Solution**: Add penalty for large weights!

---

### **L2 Regularization (Ridge)**

```
Loss = MSE + α × Σwᵢ²

α = regularization strength
Σwᵢ² = sum of squared weights
```

**Effect:**
- Keeps weights small
- Distributes weight across features
- Smoother model
- Better generalization

**When to use:**
- Many correlated features
- All features potentially useful
- Default choice

---

### **L1 Regularization (Lasso)**

```
Loss = MSE + α × Σ|wᵢ|

Σ|wᵢ| = sum of absolute values of weights
```

**Effect:**
- Some weights become exactly 0
- Automatic feature selection
- Sparse model
- Better interpretability

**When to use:**
- Many features, most are noise
- Want automatic feature selection
- Need interpretable model

---

### **Elastic Net**

```
Loss = MSE + α₁ × Σwᵢ² + α₂ × Σ|wᵢ|

Combines L1 and L2
```

**When to use:**
- Many correlated features
- Want some feature selection
- Best of both worlds

---

### **Choosing Regularization Strength (α)**

```
α = 0: No regularization (might overfit)
α small: Slight regularization
α large: Strong regularization (might underfit)

Use cross-validation to find best α!
```

---

## **YOUR EXERCISES**

### **Exercise 1: Linear Regression from Scratch**

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate simple dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2 * X + 5 + np.random.randn(100, 1) * 2  # y = 2x + 5 + noise

# Task 1: Implement linear regression using normal equation
def fit_linear_regression_normal(X, y):
    """
    Fit using normal equation: w = (X'X)^(-1)X'y
    
    Add bias term by adding column of 1s to X
    
    Returns: weights (including bias)
    """
    # YOUR CODE HERE
    pass

# Task 2: Make predictions
def predict(X, weights):
    """
    Make predictions: y = Xw
    Remember to add bias column to X
    """
    # YOUR CODE HERE
    pass

# Task 3: Calculate MSE
def mean_squared_error(y_true, y_pred):
    """
    MSE = (1/n) * sum((y_true - y_pred)^2)
    """
    # YOUR CODE HERE
    pass

# Task 4: Train and evaluate
weights = fit_linear_regression_normal(X, y)
y_pred = predict(X, weights)
mse = mean_squared_error(y, y_pred)

print(f"Weights: {weights.flatten()}")
print(f"Expected: [5, 2] (bias, slope)")
print(f"MSE: {mse:.3f}")

# Task 5: Plot results
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.show()
```

---

### **Exercise 2: Gradient Descent Implementation**

```python
# Use same data from Exercise 1

def fit_linear_regression_gd(X, y, learning_rate=0.01, epochs=1000):
    """
    Fit using gradient descent
    
    Parameters:
    X: features (n_samples, n_features)
    y: targets (n_samples, 1)
    learning_rate: step size
    epochs: number of iterations
    
    Returns:
    weights: final weights
    loss_history: list of MSE at each epoch
    """
    # Add bias column to X
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Initialize weights randomly
    n_features = X_with_bias.shape[1]
    weights = np.random.randn(n_features, 1) * 0.01
    
    loss_history = []
    
    for epoch in range(epochs):
        # YOUR CODE HERE
        # 1. Make predictions: y_pred = X @ weights
        # 2. Calculate error: error = y_pred - y
        # 3. Calculate gradient: gradient = (2/n) * X.T @ error
        # 4. Update weights: weights = weights - learning_rate * gradient
        # 5. Calculate and store MSE
        pass
    
    return weights, loss_history

# Task 1: Train with gradient descent
weights_gd, loss_history = fit_linear_regression_gd(X, y, learning_rate=0.01, epochs=1000)

print(f"GD Weights: {weights_gd.flatten()}")
print(f"Normal Equation Weights: {weights.flatten()}")

# Task 2: Plot loss over epochs
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training Loss')
plt.show()

# Task 3: Experiment with learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
# Train with each, plot loss histories
# What happens with too small/large learning rate?

# Task 4: Compare convergence
# How many epochs needed to get close to normal equation solution?
```

---

### **Exercise 3: Multiple Linear Regression**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Task 1: Explore the data
# - Print shape
# - Print feature names
# - Calculate correlation of each feature with target

# Task 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Task 3: Standardize features (important!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 4: Train linear regression
# Use your gradient descent implementation
# (you may need to adjust learning rate and epochs)

# Task 5: Evaluate on test set
# Calculate MSE, RMSE, R²

# Task 6: Feature importance
# Which features have largest weights?
# Remember to account for standardization!

# Task 7: Compare with sklearn
from sklearn.linear_model import LinearRegression
sklearn_model = LinearRegression()
sklearn_model.fit(X_train_scaled, y_train)
# Are weights similar?
```

---

### **Exercise 4: Sigmoid Function**

```python
def sigmoid(z):
    """
    Sigmoid function: σ(z) = 1 / (1 + e^(-z))
    """
    # YOUR CODE HERE
    pass

# Task 1: Test sigmoid
test_values = [-10, -5, -1, 0, 1, 5, 10]
for z in test_values:
    print(f"σ({z:3}) = {sigmoid(z):.4f}")

# Expected:
# σ(-10) ≈ 0.0000
# σ(-5)  ≈ 0.0067
# σ(0)   = 0.5000
# σ(5)   ≈ 0.9933
# σ(10)  ≈ 1.0000

# Task 2: Plot sigmoid
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
plt.show()

# Task 3: Derivative of sigmoid
# Mathematical fact: σ'(z) = σ(z) × (1 - σ(z))
def sigmoid_derivative(z):
    """Derivative of sigmoid"""
    # YOUR CODE HERE
    pass

# Plot sigmoid and its derivative
plt.plot(z, sigmoid(z), label='σ(z)')
plt.plot(z, sigmoid_derivative(z), label="σ'(z)")
plt.xlabel('z')
plt.legend()
plt.title('Sigmoid and Derivative')
plt.show()
```

---

### **Exercise 5: Logistic Regression from Scratch**

```python
from sklearn.datasets import make_classification

# Generate binary classification dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42, n_clusters_per_class=1)

# Task 1: Implement binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred):
    """
    BCE = -[y*log(y_pred) + (1-y)*log(1-y_pred)]
    Average over all samples
    
    Add small epsilon to avoid log(0)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # YOUR CODE HERE
    pass

# Task 2: Implement logistic regression with gradient descent
def fit_logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    """
    Train logistic regression using gradient descent
    
    Returns:
    weights: learned weights
    loss_history: BCE loss at each epoch
    """
    # Add bias
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    y = y.reshape(-1, 1)
    
    # Initialize weights
    n_features = X_with_bias.shape[1]
    weights = np.random.randn(n_features, 1) * 0.01
    
    loss_history = []
    
    for epoch in range(epochs):
        # YOUR CODE HERE
        # 1. Linear combination: z = X @ weights
        # 2. Apply sigmoid: y_pred = sigmoid(z)
        # 3. Calculate BCE loss
        # 4. Calculate gradient: gradient = (1/n) * X.T @ (y_pred - y)
        # 5. Update weights
        pass
    
    return weights, loss_history

# Task 3: Train model
weights, loss_history = fit_logistic_regression(X, y, learning_rate=0.1, epochs=1000)

# Task 4: Make predictions
def predict_proba(X, weights):
    """Return probabilities"""
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    z = X_with_bias @ weights
    return sigmoid(z)

def predict(X, weights, threshold=0.5):
    """Return class predictions"""
    probs = predict_proba(X, weights)
    return (probs > threshold).astype(int)

# Task 5: Evaluate
y_pred = predict(X, weights)
accuracy = np.mean(y_pred.flatten() == y)
print(f"Accuracy: {accuracy:.3f}")

# Task 6: Plot decision boundary
# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = predict(np.c_[xx.ravel(), yy.ravel()], weights)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()

# Task 7: Plot loss
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('Training Loss')
plt.show()
```

---

### **Exercise 6: Regularization**

```python
# Generate dataset prone to overfitting
np.random.seed(42)
n_samples = 50
n_features = 20  # More features than samples!

X = np.random.randn(n_samples, n_features)
# True weights: only first 3 features matter
true_weights = np.zeros(n_features)
true_weights[:3] = [3, 1.5, -2]
y = X @ true_weights + np.random.randn(n_samples) * 0.5

# Split data
X_train, X_test = X[:40], X[40:]
y_train, y_test = y[:40], y[40:]

# Task 1: Implement Ridge Regression (L2)
def fit_ridge_regression(X, y, alpha=1.0):
    """
    Ridge: w = (X'X + αI)^(-1)X'y
    
    I = identity matrix
    """
    # Add bias
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # YOUR CODE HERE
    # Note: don't regularize bias term!
    # Create regularization matrix with 0 for bias
    pass

# Task 2: Implement Lasso Regression (L1) - simplified
# (True Lasso requires iterative optimization)
# Just implement gradient descent with L1 penalty
def fit_lasso_regression(X, y, alpha=1.0, learning_rate=0.01, epochs=1000):
    """
    Lasso with gradient descent
    Gradient includes: ∂L/∂w + α * sign(w)
    """
    # YOUR CODE HERE
    pass

# Task 3: Try different α values
alphas = [0, 0.01, 0.1, 1.0, 10.0]

for alpha in alphas:
    weights = fit_ridge_regression(X_train, y_train, alpha=alpha)
    
    # Evaluate on test
    X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    y_pred = X_test_bias @ weights
    mse = np.mean((y_test - y_pred)**2)
    
    # Count non-zero weights (excluding bias)
    non_zero = np.sum(np.abs(weights[1:]) > 0.01)
    
    print(f"α={alpha:5.2f}: MSE={mse:.3f}, Non-zero weights={non_zero}")

# Task 4: Plot weight magnitudes for different α
# Show how regularization shrinks weights

# Task 5: Use cross-validation to find best α
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Try alphas from 0.001 to 100
# Plot CV score vs α
```

---

### **Exercise 7: Real Dataset - Breast Cancer Classification**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Task 1: Prepare data
# - Split into train/test (80/20)
# - Standardize features

# Task 2: Train logistic regression from scratch
# Use your implementation from Exercise 5

# Task 3: Train sklearn logistic regression
from sklearn.linear_model import LogisticRegression
sklearn_model = LogisticRegression()
# Compare results with your implementation

# Task 4: Try different regularization strengths
# LogisticRegression has parameter 'C' (inverse of α)
# C=1.0 is default
# Try C = [0.01, 0.1, 1.0, 10.0, 100.0]

# Task 5: For best model, create:
# - Confusion matrix
# - Classification report
# - ROC curve

# Task 6: Feature importance
# Which features most important for prediction?
# Plot top 10 features by weight magnitude
```

---

### **Exercise 8: Polynomial Regression**

```python
# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.5

# Task 1: Try linear regression
# It will underfit - plot the results

# Task 2: Create polynomial features
def create_polynomial_features(X, degree):
    """
    Create polynomial features up to given degree
    
    Example: X = [2], degree = 3
    Returns: [1, 2, 4, 8] (for [x⁰, x¹, x², x³])
    """
    # YOUR CODE HERE
    pass

# Task 3: Fit polynomial regression of different degrees
degrees = [1, 2, 3, 5, 10]

plt.figure(figsize=(15, 3))
for i, degree in enumerate(degrees):
    # Create polynomial features
    X_poly = create_polynomial_features(X, degree)
    
    # Fit linear regression on polynomial features
    weights = fit_linear_regression_normal(X_poly, y)
    
    # Predict
    y_pred = predict(X_poly, weights)
    
    # Plot
    plt.subplot(1, 5, i+1)
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, y_pred, 'r-')
    plt.title(f'Degree {degree}')
    plt.ylim(-5, 15)

plt.tight_layout()
plt.show()

# Task 4: Which degree fits best?
# Calculate MSE for each

# Task 5: What happens with degree 20?
# Overfitting!
```

---

### **Exercise 9: Comparing All Linear Models**

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Generate dataset
X, y = make_regression(n_samples=200, n_features=10, 
                       n_informative=5, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Task 1: Train all models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Lasso (α=1.0)': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0)
}

results = {}
for name, model in models.items():
    # YOUR CODE HERE
    # - Train model
    # - Predict on test
    # - Calculate MSE and R²
    # - Count non-zero weights
    # - Store in results dict
    pass

# Task 2: Create comparison table
# Print name, MSE, R², non-zero weights

# Task 3: Visualize weight distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for (name, model), ax in zip(models.items(), axes.flat):
    weights = model.coef_
    ax.bar(range(len(weights)), weights)
    ax.set_title(name)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Weight')

plt.tight_layout()
plt.show()

# Task 4: Which model generalizes best?
```

---

### **Exercise 10: Complete Pipeline**

```python
# Build production-ready linear model pipeline

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Task 1: Load and explore data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Explore:
# - Print shape, feature names
# - Check for missing values
# - Visualize distributions
# - Check correlations

# Task 2: Split data (60% train, 20% val, 20% test)

# Task 3: Preprocessing
# - Standardize features
# - Handle outliers (optional)

# Task 4: Baseline model
# Predict mean on validation set
# Calculate RMSE

# Task 5: Train Ridge regression
# Use cross-validation to find best α

# Task 6: Evaluate on validation set
# Calculate MSE, RMSE, R²
# Plot predictions vs actual

# Task 7: Feature importance analysis
# Which features most important?
# Can we remove some features?

# Task 8: Final evaluation on test set
# Never touched until now!
# Calculate all metrics

# Task 9: Residual analysis
# Plot residuals (y_true - y_pred)
# Are they normally distributed?
# Any patterns?

# Task 10: Save model
import pickle
# Save model and scaler for deployment

# Task 11: Document everything
# Write summary with:
# - Problem description
# - Data characteristics
# - Model choice and why
# - Performance metrics
# - Limitations
# - Next steps
```

---

## **KEY FORMULAS SUMMARY**

### **Linear Regression**
```
Model: ŷ = Xw + b
Loss (MSE): L = (1/n) Σ(y - ŷ)²
Normal Equation: w = (X'X)⁻¹X'y
Gradient: ∂L/∂w = -(2/n) X'(y - Xw)
Update: w = w - α × ∂L/∂w
```

### **Logistic Regression**
```
Model: P(y=1|x) = σ(Xw + b)
Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ)
Loss (BCE): L = -[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient: ∂L/∂w = (1/n) X'(ŷ - y)
```

### **Regularization**
```
Ridge (L2): L = MSE + α Σw²
Lasso (L1): L = MSE + α Σ|w|
ElasticNet: L = MSE + α₁ Σw² + α₂ Σ|w|
```

---

## **IMPORTANT NOTES**

✅ Always standardize features for gradient descent
✅ Start with simple model, add complexity if needed
✅ Use regularization to prevent overfitting
✅ Cross-validation to tune hyperparameters
✅ Linear models are baseline - beat them first!

**Master these concepts - they're the foundation of deep learning!**
