# Module 3.4: Other Important Algorithms

## **Why These Algorithms Matter**

**Complete your ML toolkit with essential algorithms.**

- **K-Nearest Neighbors (KNN)**: Simplest ML algorithm, great intuition
- **Support Vector Machines (SVM)**: Powerful for high-dimensional data
- **Naive Bayes**: Fast, works well for text classification
- **K-Means Clustering**: Most popular unsupervised learning

These algorithms appear in interviews, research papers, and production systems. Understanding them makes you a well-rounded ML practitioner.

---

## **CONCEPT 1: K-Nearest Neighbors (KNN)**

### **The Simplest ML Algorithm**

**Idea**: "You are the average of your K nearest neighbors"

**No training phase!** Just store the data.

**Prediction:**
1. Find K closest training examples to new point
2. Classification: Vote (majority class)
3. Regression: Average their values

---

### **How It Works - Classification**

```
Training data:
Red points: Class A
Blue points: Class B

New point (green): ?

K=3: Find 3 nearest neighbors
If 2 red, 1 blue → Predict Class A

K=5: Find 5 nearest neighbors
If 3 blue, 2 red → Predict Class B
```

**Visual:**
```
        B
    A       B
  B   ?   A
    A       B
        A

K=3: nearest are A, A, B → Predict A
K=5: nearest are A, A, B, B, B → Predict B
```

---

### **Distance Metrics**

**Euclidean Distance** (most common):
```
d(x, y) = √(Σ(xᵢ - yᵢ)²)

For 2D: d = √((x₁-y₁)² + (x₂-y₂)²)
```

**Manhattan Distance:**
```
d(x, y) = Σ|xᵢ - yᵢ|

Sum of absolute differences
```

**Minkowski Distance** (generalization):
```
d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)

p=1: Manhattan
p=2: Euclidean
```

---

### **Choosing K**

```
K=1: 
- Very sensitive to noise
- Overfits
- Irregular decision boundary

K=5 or K=7:
- Often good default
- Smooths noise
- Better generalization

K=N (all data):
- Underfits
- Predicts most common class
- Too simple

Rule of thumb: K = √N
Try odd numbers to avoid ties
```

---

### **KNN Strengths & Weaknesses**

**Strengths:**
✓ Simple to understand and implement
✓ No training time (lazy learning)
✓ Naturally handles multi-class
✓ Non-parametric (no assumptions about data)

**Weaknesses:**
✗ Slow prediction (must compare to all points)
✗ Memory intensive (stores all training data)
✗ Sensitive to feature scaling
✗ Curse of dimensionality (poor in high dimensions)
✗ Sensitive to irrelevant features

**Critical:** ALWAYS scale features for KNN!

---

## **CONCEPT 2: Support Vector Machines (SVM)**

### **Finding the Best Decision Boundary**

**Idea**: Find the hyperplane that best separates classes with maximum margin.

**Visual (2D):**
```
Class A (o)     |     Class B (x)
                |
    o           |           x
  o   o         |         x   x
    o           |           x
                |
         <--- margin --->

Goal: Maximize margin (distance between classes and boundary)
```

---

### **Linear SVM**

**For linearly separable data:**

```
Decision boundary: w·x + b = 0

w = weight vector (perpendicular to boundary)
b = bias
```

**Support Vectors:**
- Training points closest to decision boundary
- Only these points matter!
- Rest of data can be ignored

**Margin:**
```
Margin = 2 / ||w||

Maximize margin = Minimize ||w||
```

---

### **Soft Margin SVM**

**For non-perfectly separable data:**

Allow some misclassifications, but penalize them.

```
Minimize: ||w||² + C × Σ(errors)

C = regularization parameter
Large C: Less tolerance for errors (hard margin)
Small C: More tolerance for errors (soft margin)
```

**C controls bias-variance tradeoff:**
- Large C → Low bias, high variance (overfit)
- Small C → High bias, low variance (underfit)

---

### **Kernel Trick**

**Problem**: What if data is not linearly separable?

**Solution**: Map data to higher dimension where it IS separable!

**Example:**
```
Original space (2D):
Cannot separate with straight line

    x o x
  o   x   o
    x o x

Map to 3D:
z = x² + y²

Now separable with plane!
```

---

### **Common Kernels**

**Linear:**
```
K(x, y) = x·y
Use when: Data is linearly separable
```

**Polynomial:**
```
K(x, y) = (x·y + c)ᵈ

d = degree
Use when: Polynomial decision boundary needed
```

**RBF (Radial Basis Function) - most popular:**
```
K(x, y) = exp(-γ||x - y||²)

γ = gamma parameter
Large γ: More complex (wiggly) boundary
Small γ: Smoother boundary

Use when: Non-linear, general-purpose
```

**Sigmoid:**
```
K(x, y) = tanh(αx·y + c)
Similar to neural network activation
```

---

### **SVM Hyperparameters**

**C (regularization):**
- Controls training error vs margin
- Typical: 0.1, 1.0, 10, 100

**Gamma (for RBF):**
- Controls influence of single training example
- Large: Only nearby points matter
- Small: Far points also influence
- Typical: 0.001, 0.01, 0.1, 1

**Kernel:**
- Linear: Fast, interpretable
- RBF: Default choice, works well
- Polynomial: Specific use cases

---

### **SVM Strengths & Weaknesses**

**Strengths:**
✓ Effective in high dimensions
✓ Memory efficient (only stores support vectors)
✓ Versatile (different kernels)
✓ Works well with clear margin of separation

**Weaknesses:**
✗ Slow on large datasets (O(n³))
✗ Sensitive to feature scaling
✗ Hard to interpret
✗ Choosing kernel and parameters tricky
✗ No probability estimates (by default)

---

## **CONCEPT 3: Naive Bayes**

### **Probabilistic Classification with Independence Assumption**

**Based on Bayes' Theorem:**
```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

Predict class with highest probability
```

---

### **The "Naive" Assumption**

**Assume features are independent given the class.**

```
P(x₁, x₂, ..., xₙ | Class) = P(x₁|Class) × P(x₂|Class) × ... × P(xₙ|Class)

This is usually WRONG, but often works well anyway!
```

**Example: Email Spam**
```
Features: contains "free", contains "winner", contains "click"

Naive assumption: These are independent given spam/not spam
Reality: They're correlated
But: Model still works!
```

---

### **Types of Naive Bayes**

**1. Gaussian Naive Bayes** (continuous features)
```
Assumes features follow normal distribution

P(xᵢ|Class) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))

Calculate mean and variance for each feature per class
```

**2. Multinomial Naive Bayes** (count features)
```
For features that represent counts (word frequencies)
Used in text classification

P(xᵢ|Class) = (count of feature i in class) / (total count in class)
```

**3. Bernoulli Naive Bayes** (binary features)
```
For binary features (present/absent)

P(xᵢ|Class) = pᵢ if xᵢ = 1
            = 1-pᵢ if xᵢ = 0
```

---

### **Naive Bayes for Text Classification**

**Classic application: Spam detection**

```
Training:
For each class (spam/not spam):
  Count word frequencies
  Calculate P(word|class)

Prediction:
P(spam|email) ∝ P(spam) × P(word1|spam) × P(word2|spam) × ...

Compare P(spam|email) vs P(not spam|email)
```

**Smoothing (Laplace):**
```
Problem: What if word never appeared in training?
P(word|class) = 0 → entire probability becomes 0!

Solution: Add small constant (α=1)
P(word|class) = (count + α) / (total + α×vocabulary_size)
```

---

### **Naive Bayes Strengths & Weaknesses**

**Strengths:**
✓ Very fast (training and prediction)
✓ Works with small datasets
✓ Handles high dimensions well
✓ Simple to implement
✓ Naturally handles multi-class
✓ Great for text classification

**Weaknesses:**
✗ Independence assumption often violated
✗ Can't learn feature interactions
✗ Not great for complex relationships
✗ Probability estimates not always accurate

---

## **CONCEPT 4: K-Means Clustering**

### **Unsupervised Learning: Finding Groups**

**Goal**: Partition data into K clusters based on similarity.

**No labels!** Algorithm finds patterns on its own.

---

### **K-Means Algorithm**

```
1. Initialize: Randomly place K cluster centers

2. Repeat until convergence:
   a. Assignment step:
      Assign each point to nearest center
   
   b. Update step:
      Move each center to mean of its points

3. Stop when centers don't move (or max iterations)
```

**Visual:**
```
Initial:             After Step 1:       After Step 2:
  x x                  x x                 x x
x C x                x   x              x     x
  x x                C x x              C     x
                       x x                  x x
    x x                  x x                x x
  x C x                x   x              x   x
    x x                C x x                C x

C = cluster center
```

---

### **Distance Metric**

Usually Euclidean distance:
```
Minimize: Σ Σ ||xᵢ - μⱼ||²
          j i∈Cⱼ

For each cluster j, sum squared distances
of all points i in cluster to center μⱼ
```

---

### **Choosing K**

**Methods:**

**1. Elbow Method**
```
Plot K vs inertia (sum of squared distances)
Look for "elbow" where decrease slows

Inertia
  |  \
  |   \___
  |       \___
  +-------------> K
      ^elbow
```

**2. Silhouette Score**
```
Measures how similar point is to its cluster
vs other clusters

Score range: -1 to 1
-1: Wrong cluster
 0: On boundary
+1: Well clustered

Average silhouette score across all points
```

**3. Domain Knowledge**
```
Sometimes you know how many clusters make sense
Example: Customer segments (budget, regular, premium)
```

---

### **K-Means Strengths & Weaknesses**

**Strengths:**
✓ Simple and fast
✓ Scales to large datasets
✓ Works well with spherical clusters

**Weaknesses:**
✗ Must specify K in advance
✗ Sensitive to initialization (use k-means++)
✗ Assumes spherical clusters
✗ Sensitive to outliers
✗ Sensitive to scale (must standardize)

---

### **K-Means++**

**Better initialization:**

```
Instead of random initialization:

1. Choose first center randomly
2. For each remaining center:
   - Choose point farthest from existing centers
   - (With probability proportional to distance²)

Result: Centers spread out from start
Faster convergence, better results
```

sklearn uses K-Means++ by default!

---

### **Variations**

**Mini-Batch K-Means:**
- Use random batches instead of all data
- Much faster for large datasets
- Slight quality tradeoff

**DBSCAN** (alternative clustering):
- Density-based (not centroid-based)
- Can find arbitrary shapes
- Automatically determines K
- Good for non-spherical clusters

---

## **YOUR EXERCISES**

### **Exercise 1: KNN from Scratch**

```python
import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        # YOUR CODE HERE
        pass
    
    def predict_sample(self, x):
        """Predict single sample"""
        # YOUR CODE HERE
        # 1. Calculate distances to all training points
        # 2. Get indices of k nearest neighbors
        # 3. Get their labels
        # 4. Return most common label (mode)
        pass
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(x) for x in X])

# Test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

# Task 1: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 2: IMPORTANT - Scale features!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 3: Train and evaluate your KNN
knn = KNearestNeighbors(k=5)
knn.fit(X_train_scaled, y_train)
predictions = knn.predict(X_test_scaled)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.3f}")

# Task 4: Compare with sklearn
from sklearn.neighbors import KNeighborsClassifier
sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train_scaled, y_train)
sklearn_acc = sklearn_knn.score(X_test_scaled, y_test)
print(f"Sklearn Accuracy: {sklearn_acc:.3f}")

# Task 5: Try different K values
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNearestNeighbors(k=k)
    knn.fit(X_train_scaled, y_train)
    acc = np.mean(knn.predict(X_test_scaled) == y_test)
    print(f"K={k}: Accuracy={acc:.3f}")

# Which K is best?
```

---

### **Exercise 2: Impact of Feature Scaling on KNN**

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Create dataset with different feature scales
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42)

# Make features have very different scales
X[:, 0] = X[:, 0] * 1000  # Feature 1: large scale
X[:, 1] = X[:, 1] * 0.01  # Feature 2: small scale

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Task 1: Train KNN WITHOUT scaling
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
acc_unscaled = knn_unscaled.score(X_test, y_test)

# Task 2: Train KNN WITH scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = knn_scaled.score(X_test_scaled, y_test)

print(f"Without scaling: {acc_unscaled:.3f}")
print(f"With scaling: {acc_scaled:.3f}")
print(f"Improvement: {acc_scaled - acc_unscaled:.3f}")

# Task 3: Visualize decision boundaries
# Plot both (scaled and unscaled) side by side
# See how different they look!

# Task 4: Which feature dominates without scaling?
# Hint: Look at feature ranges
```

---

### **Exercise 3: SVM Tutorial**

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate linearly separable data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# IMPORTANT: Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 1: Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
print(f"Linear SVM Accuracy: {svm_linear.score(X_test_scaled, y_test):.3f}")

# Task 2: Visualize decision boundary and support vectors
def plot_svm_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    
    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], 
                model.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='red', linewidths=2,
                label='Support Vectors')
    
    plt.legend()
    plt.title(title)
    plt.show()

plot_svm_boundary(svm_linear, X_train_scaled, y_train, 'Linear SVM')

# Task 3: Non-linear data
X_nonlinear, y_nonlinear = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1,
    class_sep=0.5, random_state=42
)

# Make it non-linear
X_nonlinear = np.c_[X_nonlinear, (X_nonlinear[:, 0] ** 2 + X_nonlinear[:, 1] ** 2)]

X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    X_nonlinear, y_nonlinear, test_size=0.2
)

# Try linear kernel (will fail)
svm_linear_nl = SVC(kernel='linear')
svm_linear_nl.fit(X_train_nl, y_train_nl)
print(f"Linear on non-linear: {svm_linear_nl.score(X_test_nl, y_test_nl):.3f}")

# Try RBF kernel (will work)
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_rbf.fit(X_train_nl, y_train_nl)
print(f"RBF on non-linear: {svm_rbf.score(X_test_nl, y_test_nl):.3f}")

# Task 4: Tune C and gamma for RBF
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_nl, y_train_nl)

print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")
```

---

### **Exercise 4: Naive Bayes for Text Classification**

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load subset of 20 newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, 
                                 remove=('headers', 'footers', 'quotes'))

X, y = newsgroups.data, newsgroups.target

# Task 1: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 2: Convert text to features (Bag of Words)
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Training shape: {X_train_counts.shape}")

# Task 3: Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_counts, y_train)

# Task 4: Evaluate
y_pred = nb.predict(X_test_counts)
accuracy = np.mean(y_pred == y_test)
print(f"\nAccuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# Task 5: Try TF-IDF instead of counts
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
acc_tfidf = nb_tfidf.score(X_test_tfidf, y_test)

print(f"\nTF-IDF Accuracy: {acc_tfidf:.3f}")

# Task 6: Most informative features per class
feature_names = vectorizer.get_feature_names_out()

for i, category in enumerate(newsgroups.target_names):
    # Get log probabilities for this class
    log_prob = nb.feature_log_prob_[i]
    
    # Get top 10 features
    top_10_idx = np.argsort(log_prob)[-10:]
    top_10_words = [feature_names[idx] for idx in top_10_idx]
    
    print(f"\n{category}:")
    print(", ".join(top_10_words))

# Task 7: Test on custom examples
test_docs = [
    "Jesus Christ is the son of God",
    "Computer graphics and rendering algorithms",
    "Medical diagnosis and treatment",
    "Atheism and secular humanism"
]

test_counts = vectorizer.transform(test_docs)
predictions = nb.predict(test_counts)

for doc, pred in zip(test_docs, predictions):
    print(f"\n'{doc}'")
    print(f"Predicted: {newsgroups.target_names[pred]}")
```

---

### **Exercise 5: K-Means from Scratch**

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centers = None
        self.labels = None
    
    def initialize_centers(self, X):
        """Randomly initialize k centers from data"""
        # YOUR CODE HERE
        pass
    
    def assign_clusters(self, X):
        """Assign each point to nearest center"""
        # YOUR CODE HERE
        # For each point:
        #   Calculate distance to each center
        #   Assign to nearest center
        # Return array of cluster labels
        pass
    
    def update_centers(self, X, labels):
        """Update centers to mean of assigned points"""
        # YOUR CODE HERE
        # For each cluster:
        #   Calculate mean of all points in cluster
        # Return new centers
        pass
    
    def fit(self, X):
        """Run K-means algorithm"""
        # Initialize centers
        self.centers = self.initialize_centers(X)
        
        for iteration in range(self.max_iters):
            # YOUR CODE HERE
            # 1. Assign clusters
            # 2. Update centers
            # 3. Check convergence (optional)
            pass
        
        self.labels = self.assign_clusters(X)
        return self
    
    def predict(self, X):
        """Assign new points to nearest cluster"""
        return self.assign_clusters(X)

# Generate sample data
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Task 1: Visualize data
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Data')
plt.show()

# Task 2: Run your K-Means
kmeans = KMeans(k=4)
kmeans.fit(X)

# Task 3: Visualize results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, alpha=0.5)
plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], 
            c='red', marker='X', s=200, edgecolors='black')
plt.title('K-Means Clustering')
plt.show()

# Task 4: Compare with sklearn
from sklearn.cluster import KMeans as SklearnKMeans
sklearn_kmeans = SklearnKMeans(n_clusters=4, random_state=42)
sklearn_labels = sklearn_kmeans.fit_predict(X)

# Are results similar?

# Task 5: Elbow method
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = SklearnKMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# What's the optimal K?
```

---

### **Exercise 6: Comparing All Algorithms**

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Task: Compare all algorithms
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM (Linear)': SVC(kernel='linear'),
    'SVM (RBF)': SVC(kernel='rbf'),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Use scaled data for distance-based methods
    if name in ['KNN', 'SVM (Linear)', 'SVM (RBF)']:
        scores = cross_val_score(model, X_scaled, y, cv=5)
    else:
        scores = cross_val_score(model, X, y, cv=5)
    
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    
    print(f"{name:15s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Plot comparison
import matplotlib.pyplot as plt

names = list(results.keys())
means = [results[name]['mean'] for name in names]
stds = [results[name]['std'] for name in names]

plt.figure(figsize=(10, 6))
plt.barh(names, means, xerr=stds)
plt.xlabel('Accuracy')
plt.title('Algorithm Comparison')
plt.tight_layout()
plt.show()

# Questions:
# 1. Which algorithm performs best?
# 2. Which is most consistent (lowest std)?
# 3. Which would you choose for production?
```

---

### **Exercise 7: Clustering Customer Data**

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# Simulate customer data
np.random.seed(42)
n_customers = 500

# Create features with different characteristics
data = {
    'age': np.random.randint(18, 70, n_customers),
    'income': np.random.lognormal(10.5, 0.5, n_customers),
    'spending_score': np.random.randint(1, 100, n_customers),
    'visits_per_month': np.random.poisson(5, n_customers)
}

df = pd.DataFrame(data)

# Task 1: Explore data
print(df.describe())

# Task 2: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Task 3: Find optimal K using elbow and silhouette
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot both metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

plt.show()

# Task 4: Choose best K and fit
best_k = 4  # Choose based on plots
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Task 5: Analyze clusters
for cluster in range(best_k):
    print(f"\nCluster {cluster}:")
    print(df[df['cluster'] == cluster].describe())

# Task 6: Visualize clusters (2D projection)
# Use first 2 principal components or any 2 features
plt.scatter(df['age'], df['income'], c=df['cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()

# Task 7: Name the clusters
# Based on characteristics, give meaningful names
# Example: "Young High Spenders", "Frequent Visitors", etc.
```

---

### **Exercise 8: SVM vs Neural Network**

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Generate non-linear dataset
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 1: SVM with RBF kernel
svm = SVC(kernel='rbf', gamma=1.0, C=1.0)
svm.fit(X_train_scaled, y_train)
svm_acc = svm.score(X_test_scaled, y_test)

# Task 2: Neural Network
nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
nn.fit(X_train_scaled, y_train)
nn_acc = nn.score(X_test_scaled, y_test)

print(f"SVM Accuracy: {svm_acc:.3f}")
print(f"Neural Network Accuracy: {nn_acc:.3f}")

# Task 3: Visualize decision boundaries
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.title(title)
    plt.show()

plot_decision_boundary(svm, X_train_scaled, y_train, 'SVM Decision Boundary')
plot_decision_boundary(nn, X_train_scaled, y_train, 'Neural Network Decision Boundary')

# Task 4: Training time comparison
import time

# SVM
start = time.time()
svm.fit(X_train_scaled, y_train)
svm_time = time.time() - start

# Neural Network
start = time.time()
nn.fit(X_train_scaled, y_train)
nn_time = time.time() - start

print(f"\nSVM training time: {svm_time:.4f}s")
print(f"NN training time: {nn_time:.4f}s")
```

---

### **Exercise 9: Ensemble of Different Algorithms**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 1: Train individual models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='rbf', probability=True)  # probability=True for soft voting
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
nb = GaussianNB()

models = [
    ('KNN', knn),
    ('SVM', svm),
    ('Decision Tree', dt),
    ('Naive Bayes', nb)
]

# Evaluate individually
for name, model in models:
    if name in ['KNN', 'SVM']:
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
    print(f"{name}: {score:.3f}")

# Task 2: Hard voting ensemble
voting_hard = VotingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf')),
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('nb', GaussianNB())
    ],
    voting='hard'
)

# Note: Need to handle scaling properly for mixed models
# For simplicity, using scaled data
voting_hard.fit(X_train_scaled, y_train)
hard_score = voting_hard.score(X_test_scaled, y_test)
print(f"\nHard Voting Ensemble: {hard_score:.3f}")

# Task 3: Soft voting ensemble
voting_soft = VotingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True)),
        ('nb', GaussianNB())
    ],
    voting='soft'
)

voting_soft.fit(X_train_scaled, y_train)
soft_score = voting_soft.score(X_test_scaled, y_test)
print(f"Soft Voting Ensemble: {soft_score:.3f}")

# Does ensemble beat individual models?
```

---

### **Exercise 10: Complete ML Pipeline with All Algorithms**

```python
# Choose a dataset and build complete pipeline
# Try all algorithms learned so far

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Task 1: Load and explore data
wine = load_wine()
X, y = wine.data, wine.target

# Explore:
# - Features
# - Target distribution
# - Missing values
# - Correlations

# Task 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Task 3: Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 4: Define all models
models = {
    # Distance-based (need scaling)
    'KNN': (KNeighborsClassifier(n_neighbors=5), True),
    'SVM (Linear)': (SVC(kernel='linear'), True),
    'SVM (RBF)': (SVC(kernel='rbf'), True),
    
    # Probabilistic
    'Naive Bayes': (GaussianNB(), False),
    'Logistic Regression': (LogisticRegression(max_iter=1000), True),
    
    # Tree-based (don't need scaling)
    'Decision Tree': (DecisionTreeClassifier(max_depth=10, random_state=42), False),
    'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    'XGBoost': (xgb.XGBClassifier(n_estimators=100, random_state=42), False)
}

# Task 5: Train and evaluate all
results = {}

for name, (model, needs_scaling) in models.items():
    if needs_scaling:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        model.fit(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
    
    results[name] = {
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'test_score': test_score
    }
    
    print(f"{name:20s}: CV={scores.mean():.3f}±{scores.std():.3f}, Test={test_score:.3f}")

# Task 6: Visualize results

# Task 7: Select best model and tune hyperparameters

# Task 8: Final evaluation with confusion matrix

# Task 9: Feature importance (for applicable models)

# Task 10: Document findings and recommendations
```

---

## **ALGORITHM SELECTION GUIDE**

### **When to Use What**

```
KNN:
✓ Small dataset
✓ Low dimensions
✓ Irregular decision boundary
✗ High dimensions
✗ Large dataset
✗ Real-time predictions needed

SVM:
✓ High dimensions
✓ Clear margin between classes
✓ Small to medium datasets
✗ Very large datasets
✗ Need probability estimates
✗ Interpretability important

Naive Bayes:
✓ Text classification
✓ Need fast training/prediction
✓ Small dataset
✓ Independent features (or close enough)
✗ Feature interactions important
✗ Need high accuracy

K-Means:
✓ Unsupervised learning
✓ Need customer segmentation
✓ Spherical clusters
✗ Arbitrary cluster shapes
✗ Different cluster sizes
✗ Know true number of groups
```

---

## **KEY TAKEAWAYS**

✅ KNN: Simple but effective, MUST scale features
✅ SVM: Powerful for high-dimensional data, kernel trick for non-linear
✅ Naive Bayes: Fast, great for text, independence assumption
✅ K-Means: Standard clustering algorithm, choose K wisely

✅ Always scale features for distance-based methods
✅ Try multiple algorithms, compare results
✅ No single "best" algorithm - depends on data
✅ Ensemble methods often beat individual algorithms

**You now have a complete ML algorithm toolkit!**
