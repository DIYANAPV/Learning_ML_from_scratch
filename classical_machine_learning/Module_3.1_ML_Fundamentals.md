# Module 3.1: ML Fundamentals

## **What is Machine Learning?**

**Machine Learning = Using data to automatically learn patterns and make predictions.**

Traditional Programming:
```
Rules + Data → Output
```

Machine Learning:
```
Data + Output → Rules (Model)
```

**Example:**
- Traditional: Write rules to detect spam emails manually
- ML: Show examples of spam/not-spam, let algorithm learn the pattern

---

## **CONCEPT 1: Types of Machine Learning**

### **1. Supervised Learning**

You have **labeled data** (input + correct output).

**Goal**: Learn function that maps input → output

**Types:**

**Classification** (predicting categories):
```
Email → Spam or Not Spam
Image → Cat, Dog, or Bird
Patient → Disease or Healthy
```

**Regression** (predicting continuous values):
```
House features → Price
Temperature/Humidity → Ice cream sales
User behavior → Time spent on website
```

**Examples:**
- Spam filter
- Image recognition
- Stock price prediction
- Medical diagnosis
- Recommendation systems

---

### **2. Unsupervised Learning**

You have **unlabeled data** (input only, no outputs).

**Goal**: Find hidden patterns or structure

**Types:**

**Clustering** (grouping similar items):
```
Customer data → Customer segments
Documents → Topic groups
Genes → Related gene groups
```

**Dimensionality Reduction** (compress data):
```
1000 features → 10 features (keeping important info)
Used for visualization, noise reduction
```

**Examples:**
- Customer segmentation
- Anomaly detection
- Data compression
- Topic modeling

---

### **3. Reinforcement Learning**

Learn by **trial and error** with rewards.

**Components:**
- Agent: The learner
- Environment: What agent interacts with
- Actions: What agent can do
- Rewards: Feedback (+/-)

**Examples:**
- Game playing (AlphaGo, Chess AI)
- Robot control
- Self-driving cars
- RLHF for LLMs (later module)

**We'll focus on Supervised Learning first (most common in industry).**

---

## **CONCEPT 2: The ML Workflow**

### **Standard Process**

```
1. Problem Definition
   ↓
2. Data Collection
   ↓
3. Data Exploration & Cleaning
   ↓
4. Feature Engineering
   ↓
5. Train/Test Split
   ↓
6. Model Selection & Training
   ↓
7. Model Evaluation
   ↓
8. Hyperparameter Tuning
   ↓
9. Final Evaluation
   ↓
10. Deployment
```

**We'll learn each step!**

---

## **CONCEPT 3: Training, Validation, Test Sets**

### **Why Split Data?**

**Problem**: If you test on same data you trained on, you're cheating!
- Model memorizes training data
- Performs poorly on new, unseen data

**Solution**: Split data into separate sets

---

### **Three-Way Split**

```
All Data (100%)
│
├─ Training Set (60-80%)
│  └─ Used to train model
│
├─ Validation Set (10-20%)
│  └─ Used to tune hyperparameters
│  └─ Used during training to check progress
│
└─ Test Set (10-20%)
   └─ ONLY used at the very end
   └─ Final performance evaluation
   └─ Never seen during training
```

**Common Splits:**
- 70% train, 15% validation, 15% test
- 80% train, 10% validation, 10% test
- 60% train, 20% validation, 20% test

---

### **Implementation**

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 of 0.8 = 0.2 of total
)

# Now you have:
# X_train, y_train (60%)
# X_val, y_val (20%)
# X_test, y_test (20%)
```

**Important:**
- `random_state=42` ensures reproducibility
- Shuffle data before splitting (usually done by default)
- For time series: don't shuffle! Split chronologically

---

## **CONCEPT 4: Overfitting vs Underfitting**

### **The Central Challenge of ML**

**Underfitting** (High Bias):
- Model too simple
- Doesn't capture patterns in data
- Poor on training AND test data

**Good Fit**:
- Model captures patterns
- Generalizes to new data
- Good on both training and test data

**Overfitting** (High Variance):
- Model too complex
- Memorizes training data (including noise)
- Great on training, poor on test data

---

### **Visual Example**

Fitting polynomial to data points:

```
Underfitting (degree 1):
    y = a + bx
    Just a straight line, misses the curve

Good Fit (degree 3):
    y = a + bx + cx² + dx³
    Captures the pattern

Overfitting (degree 20):
    y = a + bx + cx² + ... + zx²⁰
    Passes through every point, wiggly nonsense
```

---

### **How to Detect**

```python
# Training error much lower than validation error = OVERFITTING
train_score = 0.95
val_score = 0.65
# Model memorized training data!

# Both errors high = UNDERFITTING
train_score = 0.60
val_score = 0.58
# Model too simple!

# Both errors similar and low = GOOD
train_score = 0.88
val_score = 0.85
# Model generalizes well!
```

---

### **How to Fix**

**Overfitting:**
- ✅ Get more training data
- ✅ Simplify model (fewer features, less complexity)
- ✅ Add regularization (penalty for complexity)
- ✅ Use dropout (neural networks)
- ✅ Early stopping

**Underfitting:**
- ✅ Use more complex model
- ✅ Add more features
- ✅ Reduce regularization
- ✅ Train longer

---

## **CONCEPT 5: Bias-Variance Tradeoff**

### **Two Sources of Error**

**Bias** (error from wrong assumptions):
- Too simple model
- Can't capture true pattern
- Systematic errors
- → Underfitting

**Variance** (error from sensitivity to training data):
- Too complex model
- Fits noise in training data
- Model changes a lot with different training data
- → Overfitting

**Goal**: Balance bias and variance!

```
Total Error = Bias² + Variance + Irreducible Error

Sweet spot: Minimize (Bias² + Variance)
```

---

## **CONCEPT 6: Cross-Validation**

### **Problem with Single Split**

- Results depend on which data points end up in train/val/test
- Might get lucky or unlucky with split

### **Solution: K-Fold Cross-Validation**

**Process:**
1. Split data into K equal parts (folds)
2. Train K times, each time using different fold as validation
3. Average the K validation scores

```
5-Fold Example:

Fold 1: [Val][Train][Train][Train][Train]
Fold 2: [Train][Val][Train][Train][Train]
Fold 3: [Train][Train][Val][Train][Train]
Fold 4: [Train][Train][Train][Val][Train]
Fold 5: [Train][Train][Train][Train][Val]

Final score = average of 5 validation scores
```

**Benefits:**
- Every data point used for validation once
- More robust estimate of performance
- Better use of limited data

**Common:** K=5 or K=10

---

### **Implementation**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
```

**When to use:**
- Small datasets (need to maximize training data)
- Robust model evaluation
- Hyperparameter tuning

**When NOT to use:**
- Very large datasets (too slow)
- Time series (need time-based split)

---

## **CONCEPT 7: Evaluation Metrics**

### **Classification Metrics**

**1. Accuracy**
```
Accuracy = (Correct Predictions) / (Total Predictions)

Example: 90 correct out of 100 = 0.90 or 90%
```

**Problem with Accuracy:**
- Misleading on imbalanced data!
- If 95% of emails are not spam:
  - Model that always predicts "not spam" = 95% accurate
  - But useless!

---

**2. Confusion Matrix**

```
                Predicted
                Pos    Neg
Actual  Pos     TP     FN
        Neg     FP     TN

TP = True Positive (correctly predicted positive)
TN = True Negative (correctly predicted negative)
FP = False Positive (wrongly predicted positive) - Type I error
FN = False Negative (wrongly predicted negative) - Type II error
```

**Example: Spam Detection**
```
                Predicted
                Spam   Not Spam
Actual  Spam     85      15      (100 spam emails)
        Not      10     890      (900 not spam)

TP = 85, FN = 15
FP = 10, TN = 890
```

---

**3. Precision**
```
Precision = TP / (TP + FP)

"Of all positive predictions, how many were correct?"

Spam example: 85 / (85 + 10) = 0.894 or 89.4%
```

**When it matters:**
- Cost of false positives is high
- Example: Medical diagnosis (don't want to scare healthy people)

---

**4. Recall (Sensitivity)**
```
Recall = TP / (TP + FN)

"Of all actual positives, how many did we catch?"

Spam example: 85 / (85 + 15) = 0.85 or 85%
```

**When it matters:**
- Cost of false negatives is high
- Example: Disease detection (don't want to miss sick people)

---

**5. F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Harmonic mean of precision and recall
Balances both metrics

Spam example: 2 × (0.894 × 0.85) / (0.894 + 0.85) = 0.871
```

**When to use:**
- Imbalanced classes
- Need balance between precision and recall

---

**6. ROC-AUC**
```
ROC = Receiver Operating Characteristic
AUC = Area Under Curve

Plots: True Positive Rate vs False Positive Rate
at different probability thresholds

AUC = 1.0: Perfect classifier
AUC = 0.5: Random guessing
AUC < 0.5: Worse than random (you did something wrong!)
```

**When to use:**
- Comparing models
- Threshold-independent evaluation
- Binary classification

---

### **Regression Metrics**

**1. Mean Absolute Error (MAE)**
```
MAE = (1/n) × Σ|actual - predicted|

Average absolute difference
Easy to interpret (same units as target)

Example: Predicting house prices
MAE = $50,000 means average error is $50k
```

---

**2. Mean Squared Error (MSE)**
```
MSE = (1/n) × Σ(actual - predicted)²

Squares errors (penalizes large errors more)
Not in original units

Example: MSE = 2,500,000,000
(Not intuitive!)
```

---

**3. Root Mean Squared Error (RMSE)**
```
RMSE = √MSE

Square root of MSE
Back to original units
More interpretable than MSE

Example: RMSE = $50,000
```

---

**4. R² (R-squared)**
```
R² = 1 - (SS_residual / SS_total)

Range: 0 to 1 (can be negative if model is terrible)
1 = Perfect predictions
0 = Model as good as predicting mean
Negative = Worse than predicting mean!

Example: R² = 0.85 means model explains 85% of variance
```

---

## **CONCEPT 8: Baseline Models**

### **Always Start Simple**

Before complex models, establish baseline:

**Classification:**
- Predict most common class
- Random prediction
- Simple logistic regression

**Regression:**
- Predict mean
- Predict median
- Simple linear regression

**Why?**
- Know if complex model is actually better
- Understand problem difficulty
- Quick sanity check

**Example:**
```python
# For classification
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)

# If your model doesn't beat this, something's wrong!
```

---

## **YOUR EXERCISES**

### **Exercise 1: Understanding ML Types**

For each problem, identify:
- Supervised or Unsupervised?
- If supervised: Classification or Regression?

```
1. Predicting apartment rental prices
2. Grouping customers by purchasing behavior
3. Detecting fraudulent credit card transactions
4. Forecasting tomorrow's temperature
5. Identifying topics in news articles
6. Predicting if a loan will default
7. Compressing images while preserving quality
8. Estimating crop yield based on weather
9. Finding unusual network traffic patterns
10. Translating English to French
```

Write your answers with reasoning.

---

### **Exercise 2: Train-Test Split Implementation**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Task 1: Implement train-test split from scratch
def my_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets
    
    Parameters:
    X: features (numpy array)
    y: labels (numpy array)
    test_size: fraction for test set (0.0 to 1.0)
    random_state: seed for reproducibility
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    # YOUR CODE HERE
    # Hint: 
    # 1. Set random seed if provided
    # 2. Create random indices
    # 3. Shuffle indices
    # 4. Split based on test_size
    # 5. Use indices to split X and y
    pass

# Task 2: Test your implementation
X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.2, random_state=42)

# Verify:
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Test size ratio: {len(X_test)/len(X):.2f}")

# Task 3: Implement three-way split
def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=None):
    """
    Split into train, validation, and test sets
    """
    # YOUR CODE HERE
    pass

# Test it
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, val_size=0.2, test_size=0.2, random_state=42
)
```

---

### **Exercise 3: Detecting Overfitting**

```python
# Simulate model performance
np.random.seed(42)

# Model 1
model1_train_scores = [0.60, 0.62, 0.61, 0.63, 0.62]
model1_val_scores = [0.58, 0.59, 0.60, 0.59, 0.61]

# Model 2
model2_train_scores = [0.95, 0.96, 0.97, 0.98, 0.99]
model2_val_scores = [0.55, 0.53, 0.54, 0.52, 0.51]

# Model 3
model3_train_scores = [0.85, 0.86, 0.87, 0.88, 0.89]
model3_val_scores = [0.82, 0.83, 0.84, 0.84, 0.85]

# Tasks:
# 1. Calculate mean and std for each model's train and val scores

# 2. For each model, identify:
#    - Is it underfitting, overfitting, or good fit?
#    - What's the evidence?
#    - What would you do to improve it?

# 3. Plot training and validation scores over epochs
#    for all three models (3 subplots)

# 4. Which model would you choose? Why?
```

---

### **Exercise 4: Confusion Matrix Analysis**

```python
# Given predictions and true labels
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0]

# Task 1: Calculate confusion matrix by hand
# Count TP, TN, FP, FN

# Task 2: Implement these metrics from scratch
def calculate_metrics(y_true, y_pred):
    """
    Calculate all classification metrics
    
    Returns dict with:
    - accuracy
    - precision
    - recall
    - f1_score
    """
    # YOUR CODE HERE
    pass

# Task 3: Verify with sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Task 4: For this scenario:
# You're detecting cancer (1 = cancer, 0 = healthy)
# Which metric is most important? Why?
# What's the cost of FP vs FN?
```

---

### **Exercise 5: Cross-Validation Implementation**

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target

# Task 1: Implement k-fold cross-validation from scratch
def my_cross_val_score(model, X, y, cv=5):
    """
    Perform k-fold cross-validation
    
    Parameters:
    model: sklearn model (must have .fit() and .score())
    X: features
    y: labels
    cv: number of folds
    
    Returns:
    scores: list of validation scores for each fold
    """
    # YOUR CODE HERE
    # Hint:
    # 1. Split data into cv folds
    # 2. For each fold:
    #    - Use fold as validation
    #    - Use rest as training
    #    - Train model
    #    - Calculate score
    # 3. Return list of scores
    pass

# Task 2: Test your implementation
model = LogisticRegression(max_iter=200)
scores = my_cross_val_score(model, X, y, cv=5)

print(f"Scores: {scores}")
print(f"Mean: {np.mean(scores):.3f}")
print(f"Std: {np.std(scores):.3f}")

# Task 3: Compare with sklearn
from sklearn.model_selection import cross_val_score
sklearn_scores = cross_val_score(model, X, y, cv=5)

# Are they similar?
```

---

### **Exercise 6: Regression Metrics**

```python
# True values and predictions
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 190, 260, 310])

# Task 1: Implement metrics from scratch
def mean_absolute_error(y_true, y_pred):
    """MAE = average of |actual - predicted|"""
    pass

def mean_squared_error(y_true, y_pred):
    """MSE = average of (actual - predicted)²"""
    pass

def root_mean_squared_error(y_true, y_pred):
    """RMSE = sqrt(MSE)"""
    pass

def r2_score(y_true, y_pred):
    """R² = 1 - (SS_residual / SS_total)"""
    # SS_residual = sum of (actual - predicted)²
    # SS_total = sum of (actual - mean)²
    pass

# Task 2: Calculate all metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Task 3: Interpret results
# What do these values tell you about model performance?

# Task 4: Verify with sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

### **Exercise 7: Baseline Model**

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5,
                          weights=[0.9, 0.1],  # 90% class 0, 10% class 1
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 1: Train baseline (most frequent class)
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)

print(f"Baseline Accuracy: {baseline_acc:.3f}")

# Task 2: What is the baseline predicting?
# Check what class it always predicts

# Task 3: Calculate baseline precision, recall, F1
# Will they be good?

# Task 4: Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
model_acc = model.score(X_test, y_test)

print(f"Model Accuracy: {model_acc:.3f}")

# Task 5: Which model is better? Why?
# Calculate F1 score for both
# Which metric matters more for this problem?
```

---

### **Exercise 8: Visualizing Model Performance**

```python
import matplotlib.pyplot as plt

# Simulate training history
epochs = range(1, 51)
train_acc = 1 - np.exp(-np.linspace(0, 2.5, 50)) * 0.3
val_acc = 1 - np.exp(-np.linspace(0, 2, 50)) * 0.3
val_acc = val_acc + np.random.randn(50) * 0.02

# Task 1: Plot training and validation accuracy
# Mark where overfitting starts (if any)

# Task 2: Calculate the gap between train and val
# Plot the gap over epochs

# Task 3: Identify the best epoch to stop training
# (highest validation accuracy)

# Task 4: Create 2x2 subplot grid showing:
# - Top-left: Train and val accuracy
# - Top-right: Gap between train and val
# - Bottom-left: Train and val loss (simulate it)
# - Bottom-right: Text summary with best epoch, accuracies
```

---

### **Exercise 9: Real Dataset - Breast Cancer**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Task 1: Explore the data
# - How many samples?
# - How many features?
# - Class distribution?
# - Is it balanced?

# Task 2: Split data (60% train, 20% val, 20% test)

# Task 3: Train baseline model

# Task 4: Train logistic regression

# Task 5: Evaluate both models on validation set
# Calculate: accuracy, precision, recall, F1

# Task 6: Which metric is most important?
# This is cancer detection - FN vs FP cost?

# Task 7: Perform 5-fold cross-validation

# Task 8: Final evaluation on test set
# Create confusion matrix
# Write classification report

# Task 9: Visualize results
# - Confusion matrix heatmap
# - ROC curve (research how to plot it)
```

---

### **Exercise 10: Comprehensive ML Pipeline**

```python
# Build complete pipeline from scratch

# 1. Load dataset (your choice from sklearn.datasets)

# 2. Exploratory Data Analysis
#    - Check shape, dtypes
#    - Check for missing values
#    - Check class distribution
#    - Visualize feature distributions

# 3. Train-Val-Test Split

# 4. Baseline Model
#    - Train and evaluate

# 5. Simple Model (Logistic Regression)
#    - Train and evaluate
#    - Compare with baseline

# 6. Cross-Validation
#    - 5-fold CV
#    - Report mean ± std

# 7. Evaluate on Test Set
#    - Calculate all relevant metrics
#    - Create confusion matrix
#    - Visualize results

# 8. Write Summary
#    - Problem description
#    - Data characteristics
#    - Model performance
#    - Which model to deploy?
#    - Next steps for improvement

# Document everything with comments!
```

---

## **KEY TAKEAWAYS**

### **Always Remember:**
✅ Split your data (train/val/test)
✅ Start with baseline
✅ Watch for overfitting
✅ Choose metrics carefully
✅ Use cross-validation for robust evaluation
✅ Test set is sacred (use only at the end)

### **Common Mistakes:**
❌ Training and testing on same data
❌ Using accuracy on imbalanced data
❌ Not checking for overfitting
❌ Tuning on test set
❌ Ignoring baseline model

---

## **IMPORTANT NOTES**

✅ These fundamentals apply to ALL machine learning
✅ Understand these before diving into algorithms
✅ Practice identifying overfitting vs underfitting
✅ Learn to choose appropriate metrics
✅ Master train/val/test split

**Next**: We'll learn actual ML algorithms (linear regression, etc.)!
