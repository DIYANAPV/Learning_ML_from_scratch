# Module 3.3: Tree-Based Models

## **Why Tree-Based Models Matter**

**Tree-based models are the workhorses of classical ML.**

- **Powerful**: Win many Kaggle competitions
- **No feature scaling needed**: Work with raw features
- **Handle non-linear relationships**: Naturally capture complex patterns
- **Interpretable**: Can visualize decision rules
- **Industry standard**: XGBoost/LightGBM used everywhere

In tabular data (spreadsheets, databases), tree-based models often beat neural networks!

---

## **CONCEPT 1: Decision Trees**

### **What is a Decision Tree?**

A series of yes/no questions that leads to a prediction.

**Visual Example: Predicting if someone plays tennis**

```
                    Outlook?
                   /    |    \
              Sunny  Overcast  Rain
               /        |        \
          Humidity?    Yes    Windy?
           /    \              /    \
        High   Normal       True   False
         /        \          /        \
        No       Yes       No        Yes

Questions asked in order until reaching leaf (prediction)
```

---

### **How It Works - Classification**

**Algorithm:**
1. Start at root (all data)
2. Find best feature to split on
3. Split data based on that feature
4. Repeat for each branch
5. Stop when:
   - All samples in node are same class
   - Reached max depth
   - Node has too few samples
   - No more features

**Prediction:**
- Follow decision path for new sample
- Predict majority class in leaf node

---

### **How It Works - Regression**

Same process, but:
- Predict average value in leaf node
- Minimize variance instead of impurity

---

### **Example: Predicting House Prices**

```
Data:
Size | Age | Price
-----|-----|-------
1500 |  10 | 300k
2000 |   5 | 400k
1200 |  20 | 250k
1800 |   3 | 450k

Tree might learn:
        Size < 1600?
          /        \
        Yes         No
         |           |
    Age < 15?    Age < 4?
      /    \       /    \
    Yes    No    Yes    No
     |      |     |      |
   250k   300k  450k   400k
```

---

## **CONCEPT 2: Splitting Criteria**

### **How to Choose Best Split?**

**Goal**: Split data to make subgroups more "pure" (similar).

---

### **Classification: Gini Impurity**

```
Gini = 1 - Σ(pᵢ)²

pᵢ = proportion of class i in node

Perfect purity (all same class): Gini = 0
Maximum impurity (50-50 split): Gini = 0.5
```

**Example:**
```
Node with 100 samples:
- 60 class A, 40 class B

Gini = 1 - (0.6² + 0.4²)
     = 1 - (0.36 + 0.16)
     = 1 - 0.52
     = 0.48

Lower is better (more pure)
```

---

### **Classification: Entropy**

```
Entropy = -Σ(pᵢ × log₂(pᵢ))

Perfect purity: Entropy = 0
Maximum impurity: Entropy = 1 (for binary)
```

**Example:**
```
Same node: 60 class A, 40 class B

Entropy = -(0.6 × log₂(0.6) + 0.4 × log₂(0.4))
        = -(0.6 × -0.737 + 0.4 × -1.322)
        = -(-0.442 - 0.529)
        = 0.971
```

---

### **Information Gain**

```
Information Gain = Parent Impurity - Weighted Child Impurity

Choose split with highest information gain
```

**Example:**
```
Parent node: 100 samples (60 A, 40 B), Gini = 0.48

Split on feature X < 5:
- Left: 70 samples (50 A, 20 B), Gini = 0.408
- Right: 30 samples (10 A, 20 B), Gini = 0.444

Weighted child impurity = 0.7 × 0.408 + 0.3 × 0.444 = 0.419
Information gain = 0.48 - 0.419 = 0.061

Try all possible splits, choose one with highest gain
```

---

### **Regression: Variance Reduction**

```
Split to minimize variance in each group

Variance = (1/n) × Σ(yᵢ - ȳ)²
```

---

## **CONCEPT 3: Overfitting in Trees**

### **The Problem**

Trees can grow until:
- Every leaf has one sample
- Perfect predictions on training data
- Terrible on test data (overfitting!)

---

### **Preventing Overfitting: Hyperparameters**

**1. max_depth**
```
Maximum tree depth
Deeper = more complex
Too deep = overfit
Typical: 3-10
```

**2. min_samples_split**
```
Minimum samples to split a node
Higher = less splits = simpler tree
Typical: 2-20
```

**3. min_samples_leaf**
```
Minimum samples in leaf node
Prevents tiny leaves
Typical: 1-10
```

**4. max_features**
```
Number of features to consider for each split
Less = more randomness = less overfit
Used in Random Forests
```

**5. max_leaf_nodes**
```
Limit total number of leaf nodes
Another way to control complexity
```

---

## **CONCEPT 4: Random Forests**

### **Ensemble Learning: Wisdom of Crowds**

**Problem**: Single tree overfits, unstable (changes a lot with different data).

**Solution**: Train many trees, average their predictions!

---

### **Random Forest Algorithm**

```
1. Create N bootstrap samples (random sampling with replacement)
2. For each sample:
   - Train a decision tree
   - At each split, only consider random subset of features
3. Prediction:
   - Classification: majority vote
   - Regression: average
```

**Bootstrap Sample:**
```
Original: [1, 2, 3, 4, 5]
Sample 1: [1, 1, 3, 4, 5]  (some repeated, some missing)
Sample 2: [1, 2, 2, 3, 5]
Sample 3: [2, 3, 4, 4, 5]
```

---

### **Why Random Forests Work**

**1. Bagging (Bootstrap Aggregating)**
- Reduces variance
- Trees trained on different data
- Averaging smooths predictions

**2. Feature Randomness**
- Decorrelates trees
- If one feature very strong, not all trees use it
- Forces diversity

**Result:**
- More stable than single tree
- Less overfitting
- Often best out-of-box performance

---

### **Random Forest Hyperparameters**

```
n_estimators: Number of trees (more is better, diminishing returns)
max_depth: Depth of each tree
min_samples_split: Min samples to split
max_features: Features to consider per split
bootstrap: Whether to use bootstrap sampling
oob_score: Use out-of-bag samples for validation
```

**Typical settings:**
```
n_estimators=100
max_depth=None (or tune)
max_features='sqrt' for classification
max_features='1/3' for regression
```

---

## **CONCEPT 5: Gradient Boosting**

### **Boosting: Sequential Learning**

**Idea**: Train trees sequentially, each fixing previous tree's mistakes.

**Random Forest vs Gradient Boosting:**
```
Random Forest:
- Trees trained in parallel (independent)
- Each tree sees different random data
- Average predictions

Gradient Boosting:
- Trees trained sequentially (dependent)
- Each tree sees full data
- Add predictions
```

---

### **Gradient Boosting Algorithm**

```
1. Start with initial prediction (mean for regression)
2. For each tree:
   a. Calculate residuals (errors from previous predictions)
   b. Train new tree to predict residuals
   c. Add tree's predictions to ensemble
   d. Update predictions
3. Final prediction = sum of all tree predictions
```

**Example: Predicting House Price**

```
True price: 300k

Tree 1 predicts: 250k
Residual: 300k - 250k = 50k

Tree 2 trained on residual, predicts: 40k
New prediction: 250k + 40k = 290k
Residual: 300k - 290k = 10k

Tree 3 trained on residual, predicts: 8k
Final prediction: 250k + 40k + 8k = 298k
```

---

### **Learning Rate**

```
Instead of adding full tree prediction, use fraction:

prediction += learning_rate × tree_prediction

learning_rate (η) typically: 0.01 to 0.3

Lower η = need more trees, but better generalization
Higher η = fewer trees needed, but might overfit
```

---

### **Gradient Boosting Hyperparameters**

```
n_estimators: Number of trees (more is better until overfitting)
learning_rate: Step size (0.01-0.3)
max_depth: Tree depth (typically 3-10, shallow trees!)
subsample: Fraction of data for each tree (0.5-1.0)
min_samples_split: Min samples to split
min_samples_leaf: Min samples in leaf
```

**Key insight:**
- Shallow trees (max_depth=3-5)
- Many trees (100-1000)
- Low learning rate (0.01-0.1)

---

## **CONCEPT 6: XGBoost**

### **Extreme Gradient Boosting**

**Improvements over standard Gradient Boosting:**

1. **Regularization**
   - L1 and L2 on weights
   - Prevents overfitting

2. **Handling missing values**
   - Learns best direction for missing values
   - No need to impute

3. **Speed**
   - Parallel processing
   - Optimized implementation
   - Cache-aware algorithms

4. **Built-in cross-validation**

5. **Early stopping**
   - Stop training when validation score stops improving

---

### **XGBoost Key Parameters**

```python
# Basic
n_estimators=100           # Number of trees
learning_rate=0.1         # Step size
max_depth=6               # Tree depth

# Regularization
reg_alpha=0               # L1 regularization
reg_lambda=1              # L2 regularization
gamma=0                   # Min loss reduction to split

# Sampling
subsample=1.0             # Fraction of data per tree
colsample_bytree=1.0      # Fraction of features per tree

# Other
objective='reg:squarederror'  # Loss function
eval_metric='rmse'        # Evaluation metric
early_stopping_rounds=10  # Stop if no improvement
```

---

## **CONCEPT 7: LightGBM**

### **Light Gradient Boosting Machine**

**Key innovation**: Leaf-wise tree growth (vs level-wise).

**Level-wise (traditional):**
```
       Root
      /    \
    L1      L1    ← Fill this level
   /  \    /  \
  L2  L2  L2  L2  ← Then this level
```

**Leaf-wise (LightGBM):**
```
Grow leaf with highest gain first
Faster, but can overfit more easily
Need to limit max_depth
```

**Advantages:**
- Much faster than XGBoost
- Lower memory usage
- Handles large datasets well
- Good accuracy

**Disadvantages:**
- Easier to overfit
- Need to tune max_depth carefully

---

## **CONCEPT 8: Feature Importance**

### **Understanding What Matters**

**How trees measure importance:**

**1. Split count**
- How many times feature used for splitting
- More splits = more important

**2. Gain/impurity reduction**
- Total reduction in loss from splits using this feature
- Better metric than count

**3. Permutation importance**
- Shuffle feature values
- See how much performance drops
- Drop = importance

---

### **Using Feature Importance**

```python
# After training
importances = model.feature_importances_

# Plot
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance')

# Select top K features
top_k = np.argsort(importances)[-10:]  # Top 10
```

**Benefits:**
- Feature selection
- Understanding model
- Domain insights
- Debugging

---

## **YOUR EXERCISES**

### **Exercise 1: Decision Tree from Scratch (Simplified)**

```python
import numpy as np

class SimpleDecisionTree:
    """Simple decision tree for binary classification"""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def gini_impurity(self, y):
        """
        Calculate Gini impurity
        Gini = 1 - Σ(p_i)²
        """
        # YOUR CODE HERE
        pass
    
    def split_data(self, X, y, feature_idx, threshold):
        """
        Split data based on feature and threshold
        
        Returns:
        left_X, left_y, right_X, right_y
        """
        # YOUR CODE HERE
        pass
    
    def find_best_split(self, X, y):
        """
        Find best feature and threshold to split on
        
        Try all features, all unique values as thresholds
        Return feature_idx, threshold with highest information gain
        """
        # YOUR CODE HERE
        # For each feature:
        #   For each unique value as threshold:
        #     Calculate information gain
        # Return best split
        pass
    
    def build_tree(self, X, y, depth=0):
        """
        Recursively build tree
        
        Returns: dict representing tree node
        {
            'feature': feature_idx,
            'threshold': value,
            'left': left_subtree,
            'right': right_subtree,
            'value': predicted_class (for leaf)
        }
        """
        # YOUR CODE HERE
        # Base cases:
        # - If all same class: return leaf
        # - If max_depth reached: return leaf (majority class)
        # 
        # Recursive case:
        # - Find best split
        # - Split data
        # - Build left and right subtrees
        pass
    
    def fit(self, X, y):
        """Train the tree"""
        self.tree = self.build_tree(X, y)
    
    def predict_sample(self, x, tree):
        """Predict single sample"""
        # YOUR CODE HERE
        # If leaf: return value
        # Otherwise: go left or right based on feature
        pass
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(x, self.tree) for x in X])

# Test your tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=4, n_redundant=0,
                          n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree = SimpleDecisionTree(max_depth=3)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.3f}")
```

---

### **Exercise 2: Understanding Tree Splits**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris
iris = load_iris()
X, y = iris.data[:, [0, 1]], iris.target  # Use only 2 features for visualization

# Task 1: Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Task 2: Visualize decision boundary
def plot_decision_boundary(model, X, y):
    """Plot decision boundary"""
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('Decision Tree Boundary')
    plt.show()

plot_decision_boundary(tree, X, y)

# Task 3: Visualize tree structure
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(tree, feature_names=iris.feature_names[:2], 
          class_names=iris.target_names, filled=True)
plt.show()

# Task 4: Try different max_depths
# Plot decision boundaries for depth = 1, 2, 3, 5, 10
# What happens as depth increases?

# Task 5: Calculate training and test accuracy for each depth
# Plot depth vs accuracy
# When does overfitting start?
```

---

### **Exercise 3: Random Forest Implementation**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 1: Train Random Forest with different n_estimators
n_estimators_list = [1, 5, 10, 50, 100, 200]

train_scores = []
test_scores = []

for n in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"n={n:3d}: Train={train_score:.3f}, Test={test_score:.3f}")

# Task 2: Plot n_estimators vs accuracy
plt.plot(n_estimators_list, train_scores, label='Train')
plt.plot(n_estimators_list, test_scores, label='Test')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Random Forest: Trees vs Accuracy')
plt.show()

# Task 3: Feature importance
best_rf = RandomForestClassifier(n_estimators=100, random_state=42)
best_rf.fit(X_train, y_train)

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices[:10]])
plt.yticks(range(10), [data.feature_names[i] for i in indices[:10]])
plt.xlabel('Importance')
plt.title('Top 10 Important Features')
plt.show()

# Task 4: Train with only top 10 features
# Does accuracy change?

# Task 5: Hyperparameter tuning
# Try different combinations:
# - max_depth: [None, 5, 10, 20]
# - min_samples_split: [2, 5, 10]
# - max_features: ['sqrt', 'log2', None]
# Use cross-validation to find best
```

---

### **Exercise 4: Gradient Boosting from Scratch (Simplified)**

```python
from sklearn.tree import DecisionTreeRegressor

class SimpleGradientBoosting:
    """Simple gradient boosting for regression"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X, y):
        """Train gradient boosting model"""
        # Initial prediction: mean
        self.initial_prediction = np.mean(y)
        
        # Start with initial predictions
        predictions = np.full(len(y), self.initial_prediction)
        
        for i in range(self.n_estimators):
            # YOUR CODE HERE
            # 1. Calculate residuals: residuals = y - predictions
            # 2. Train tree on residuals
            # 3. Update predictions: predictions += learning_rate * tree.predict(X)
            # 4. Store tree
            pass
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        # YOUR CODE HERE
        # 1. Start with initial prediction
        # 2. Add each tree's prediction (scaled by learning_rate)
        pass

# Test
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Task 1: Train your gradient boosting
gb = SimpleGradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)
predictions = gb.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse:.3f}")

# Task 2: Compare with sklearn
from sklearn.ensemble import GradientBoostingRegressor
sklearn_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                       max_depth=3, random_state=42)
sklearn_gb.fit(X_train, y_train)
sklearn_pred = sklearn_gb.predict(X_test)
sklearn_mse = mean_squared_error(y_test, sklearn_pred)
print(f"Sklearn MSE: {sklearn_mse:.3f}")

# Task 3: Experiment with learning rate
# Try: 0.01, 0.05, 0.1, 0.5
# What happens to training speed and final performance?
```

---

### **Exercise 5: XGBoost Tutorial**

```python
import xgboost as xgb
from sklearn.datasets import load_boston  # or use california housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load data
# Note: load_boston is deprecated, use fetch_california_housing instead
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 1: Train basic XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Task 2: Feature importance
xgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.show()

# Task 3: Learning curves
# Train with different n_estimators, plot performance
n_estimators_list = [10, 50, 100, 200, 500]
train_rmse = []
test_rmse = []

for n in n_estimators_list:
    model = xgb.XGBRegressor(n_estimators=n, learning_rate=0.1, 
                             max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, test_pred)))

plt.plot(n_estimators_list, train_rmse, label='Train')
plt.plot(n_estimators_list, test_rmse, label='Test')
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# Task 4: Early stopping
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")

# Task 5: Hyperparameter tuning with GridSearch
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_:.3f}")
```

---

### **Exercise 6: Model Comparison**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Task: Compare all models with cross-validation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Plot comparison
names = list(results.keys())
means = [results[name]['mean'] for name in names]
stds = [results[name]['std'] for name in names]

plt.figure(figsize=(10, 6))
plt.barh(names, means, xerr=stds)
plt.xlabel('Accuracy')
plt.title('Model Comparison (5-Fold CV)')
plt.show()

# Which model performs best?
# Which is fastest to train?
# Which is most interpretable?
```

---

### **Exercise 7: Overfitting Analysis**

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Task 1: Decision Tree - vary max_depth
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

plt.plot(depths, train_scores, label='Train')
plt.plot(depths, test_scores, label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Decision Tree: Depth vs Accuracy')
plt.show()

# At what depth does overfitting start?

# Task 2: Random Forest - does it overfit less?
# Repeat same analysis for Random Forest

# Task 3: Compare final model complexities
# - Decision Tree at best depth
# - Random Forest
# - XGBoost
# Which generalizes best?
```

---

### **Exercise 8: Real-World Dataset - Credit Card Fraud**

```python
# Note: You'll need to download this dataset or use another
# For now, let's simulate imbalanced data

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Simulate imbalanced fraud dataset
X, y = make_classification(n_samples=10000, n_features=30, n_informative=20,
                          n_redundant=5, weights=[0.98, 0.02],  # 2% fraud
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      stratify=y, random_state=42)

print(f"Training set fraud rate: {y_train.mean():.3f}")
print(f"Test set fraud rate: {y_test.mean():.3f}")

# Task 1: Baseline (predict always "not fraud")
# What's the accuracy? Is it meaningful?

# Task 2: Train XGBoost
# Use scale_pos_weight to handle imbalance
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42
)

model.fit(X_train, y_train)

# Task 3: Evaluate properly
# - Don't use accuracy!
# - Use precision, recall, F1
# - Use ROC-AUC
# - Confusion matrix

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Task 4: Adjust threshold
# Default threshold = 0.5
# Try different thresholds to balance precision/recall
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for threshold in thresholds:
    y_pred_custom = (y_pred_proba > threshold).astype(int)
    # Calculate precision, recall for each
    # Which threshold is best for fraud detection?
```

---

### **Exercise 9: Feature Engineering with Trees**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Task 1: Baseline XGBoost
baseline = xgb.XGBRegressor(n_estimators=100, random_state=42)
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
print(f"Baseline R²: {baseline_score:.3f}")

# Task 2: Add polynomial features
# Create interactions between important features
# (Find important features first!)

# Task 3: Add binned features
# Bin continuous features into categories
# Example: age → young/medium/old

# Task 4: Add domain knowledge features
# Example: rooms_per_household = total_rooms / households

# Task 5: Compare all models
# Does feature engineering help tree-based models?
# (Spoiler: usually not as much as for linear models!)
```

---

### **Exercise 10: Complete Kaggle-Style Pipeline**

```python
# Build production-ready tree-based model pipeline

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd

# Task 1: Load and explore
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Explore:
# - Missing values?
# - Distributions
# - Correlations
# - Outliers

# Task 2: Train-validation-test split

# Task 3: Handle missing values (if any)
# Trees can handle missing values, but make it explicit

# Task 4: Baseline model
# Simple model for comparison

# Task 5: XGBoost with default parameters

# Task 6: Hyperparameter tuning
# Use RandomizedSearchCV for efficiency

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

# Task 7: Feature importance analysis
# - Plot top 20 features
# - Try training with only top 10
# - Does it hurt performance?

# Task 8: Learning curves
# Plot training/validation error vs:
# - Number of trees
# - Max depth
# - Learning rate

# Task 9: Final evaluation on test set

# Task 10: Model interpretation
# - Feature importance
# - Partial dependence plots (if time)
# - SHAP values (advanced, optional)

# Task 11: Save model
import pickle
# Save best model for deployment

# Task 12: Write documentation
# - Model performance
# - Key findings
# - Deployment recommendations
```

---

## **KEY CONCEPTS SUMMARY**

### **Decision Trees**
- Series of if-then rules
- Gini or Entropy for splits
- Prone to overfitting
- Control with max_depth, min_samples_split

### **Random Forest**
- Ensemble of trees
- Bootstrap + feature randomness
- Reduces variance
- More stable than single tree

### **Gradient Boosting**
- Sequential learning
- Each tree fixes previous mistakes
- Powerful but can overfit
- Use low learning rate + many trees

### **XGBoost/LightGBM**
- Optimized implementations
- Regularization built-in
- Handle missing values
- Industry standard for tabular data

---

## **WHEN TO USE WHAT**

```
Decision Tree:
✓ Need interpretability
✓ Quick baseline
✗ Final production model (too simple)

Random Forest:
✓ Good out-of-box performance
✓ Robust, less tuning needed
✓ Feature importance
✗ Very large datasets (slow)

Gradient Boosting/XGBoost:
✓ Best performance on tabular data
✓ Kaggle competitions
✓ Production systems
✗ Need lots of tuning
✗ Can overfit easily

LightGBM:
✓ Very large datasets
✓ Speed is critical
✓ Same accuracy as XGBoost
✗ Easier to overfit than XGBoost
```

---

## **IMPORTANT NOTES**

✅ Trees don't need feature scaling
✅ Start with Random Forest (good baseline)
✅ XGBoost for best performance
✅ Use cross-validation for tuning
✅ Watch for overfitting
✅ Feature importance is valuable
✅ Trees excel at tabular data

**Master tree-based models - they're the go-to for structured data!**
