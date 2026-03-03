# Module 2.3: Data Visualization

## **Why Visualization Matters**

**You can't understand data without seeing it.**

- **Spot patterns**: Trends, clusters, outliers jump out visually
- **Debug models**: Visualize predictions vs reality
- **Communicate**: Charts explain better than tables
- **EDA**: Exploratory Data Analysis is visual first

In ML, visualization helps you understand your data before modeling and diagnose issues after.

---

## **CONCEPT 1: Matplotlib Basics**

### **The Foundation of Python Visualization**

Matplotlib is the base - everything else builds on it.

```python
import matplotlib.pyplot as plt
import numpy as np

# Basic plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.show()

# With labels and title
plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('My First Plot')
plt.show()

# Multiple lines
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.legend()
plt.show()
```

### **Customizing Plots**

```python
# Line style and color
plt.plot(x, y, color='red', linestyle='--', linewidth=2)
plt.plot(x, y, 'r--')  # Shorthand: color + style

# Colors: 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
# Styles: '-', '--', '-.', ':', ''

# Markers
plt.plot(x, y, marker='o')  # Circle markers
plt.plot(x, y, 'o-')        # Line with circles

# Markers: 'o', 's', '^', 'v', '*', '+', 'x', 'D'

# Grid
plt.grid(True)
plt.grid(True, alpha=0.3)  # Transparent grid

# Limits
plt.xlim(0, 10)
plt.ylim(-5, 5)
```

### **Figure and Subplots**

```python
# Create figure and axis
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Title')

# Multiple subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(x, y1)
ax1.set_title('Plot 1')
ax2.plot(x, y2)
ax2.set_title('Plot 2')
plt.tight_layout()  # Prevent overlap

# Grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].plot(x, y1)
axes[0, 1].scatter(x, y2)
axes[1, 0].bar([1, 2, 3], [4, 5, 6])
axes[1, 1].hist(np.random.randn(1000))
```

---

## **CONCEPT 2: Common Plot Types**

### **1. Line Plot** (trends over time)

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Use case: Stock prices, model loss over epochs
```

### **2. Scatter Plot** (relationships between variables)

```python
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5

plt.scatter(x, y, alpha=0.5)
plt.title('Correlation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# With color coding
colors = ['red' if val > 0 else 'blue' for val in y]
plt.scatter(x, y, c=colors, alpha=0.5)

# Use case: Feature correlation, clustering results
```

### **3. Bar Plot** (comparing categories)

```python
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.bar(categories, values)
plt.title('Category Comparison')
plt.ylabel('Value')
plt.show()

# Horizontal bars
plt.barh(categories, values)

# Use case: Model comparison, feature importance
```

### **4. Histogram** (distribution of data)

```python
data = np.random.randn(1000)

plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Multiple distributions
plt.hist([data1, data2], bins=30, label=['Group 1', 'Group 2'], alpha=0.5)
plt.legend()

# Use case: Understanding feature distributions
```

### **5. Box Plot** (distribution summary with outliers)

```python
data = [np.random.randn(100), np.random.randn(100)+1, np.random.randn(100)+2]

plt.boxplot(data, labels=['A', 'B', 'C'])
plt.title('Distribution Comparison')
plt.ylabel('Value')
plt.show()

# Box shows: min, Q1, median, Q3, max
# Outliers shown as dots

# Use case: Comparing distributions, spotting outliers
```

### **6. Heatmap** (2D data, correlations)

```python
import numpy as np

# Create correlation matrix
data = np.random.randn(5, 5)

plt.imshow(data, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Heatmap')
plt.show()

# Use case: Correlation matrices, confusion matrices
```

---

## **CONCEPT 3: Seaborn - Statistical Visualization**

### **Why Seaborn?**

Built on Matplotlib but:
- Better default styles
- Built for statistical plots
- Works great with Pandas DataFrames

```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')  # or 'darkgrid', 'white', 'dark', 'ticks'
sns.set_palette('husl')     # Color palette
```

### **Distribution Plots**

```python
import pandas as pd

# Sample data
df = pd.DataFrame({
    'value': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

# Histogram with KDE (Kernel Density Estimate)
sns.histplot(data=df, x='value', kde=True)
plt.show()

# Just KDE
sns.kdeplot(data=df, x='value')
plt.show()

# Distribution by category
sns.histplot(data=df, x='value', hue='category', kde=True)
plt.show()
```

### **Relationship Plots**

```python
# Scatter with regression line
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
df['y'] = 2*df['x'] + df['y']*0.5

sns.regplot(data=df, x='x', y='y')
plt.show()

# Scatter colored by category
df['category'] = np.random.choice(['A', 'B'], 100)
sns.scatterplot(data=df, x='x', y='y', hue='category', style='category')
plt.show()
```

### **Categorical Plots**

```python
# Box plot
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 300),
    'value': np.random.randn(300)
})

sns.boxplot(data=df, x='category', y='value')
plt.show()

# Violin plot (like box plot but shows distribution shape)
sns.violinplot(data=df, x='category', y='value')
plt.show()

# Bar plot with confidence intervals
sns.barplot(data=df, x='category', y='value')
plt.show()

# Count plot (histogram for categories)
sns.countplot(data=df, x='category')
plt.show()
```

### **Matrix Plots**

```python
# Correlation heatmap
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
corr = df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Pair plot (scatter for all pairs)
sns.pairplot(df)
plt.show()

# With categories
df['category'] = np.random.choice(['X', 'Y'], 100)
sns.pairplot(df, hue='category')
plt.show()
```

---

## **CONCEPT 4: Visualization for ML**

### **1. Feature Distributions**

```python
# Check if features are normally distributed
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(df.columns[:6]):
    ax = axes[i//3, i%3]
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')
plt.tight_layout()
```

### **2. Feature Correlations**

```python
# Correlation matrix
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Feature Correlations')
plt.show()
```

### **3. Outlier Detection**

```python
# Box plots for each feature
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, col in enumerate(df.columns[:5]):
    axes[i].boxplot(df[col])
    axes[i].set_title(col)
plt.tight_layout()
```

### **4. Class Distribution** (for classification)

```python
# For imbalanced datasets
class_counts = df['target'].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Or pie chart
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()
```

### **5. Model Performance**

```python
# Training history
epochs = range(1, 51)
train_loss = np.exp(-np.linspace(0, 2, 50)) + np.random.randn(50)*0.02
val_loss = np.exp(-np.linspace(0, 1.8, 50)) + np.random.randn(50)*0.03

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Training')
plt.show()

# Predictions vs Actual
y_true = np.random.randn(100)
y_pred = y_true + np.random.randn(100)*0.3

plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
         'r--', label='Perfect prediction')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.title('Predictions vs Actual')
plt.show()
```

---

## **YOUR EXERCISES**

### **Exercise 1: Basic Matplotlib**

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. Create line plot
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
# Plot with red dashed line, add labels, title, grid

# 2. Create multiple lines on same plot
# Plot sin(x), cos(x), and sin(2x)
# Use different colors and styles
# Add legend

# 3. Create subplot with 2 plots side by side
# Left: sin(x), Right: cos(x)

# 4. Create 2x2 grid of different functions:
# Top-left: sin(x), Top-right: cos(x)
# Bottom-left: tan(x), Bottom-right: exp(-x)
```

---

### **Exercise 2: Scatter Plots**

```python
# Generate data with correlation
np.random.seed(42)
n = 200
x = np.random.randn(n)
y = 2*x + np.random.randn(n)*0.5
z = -x + np.random.randn(n)*0.5

# 1. Create scatter plot x vs y
# Color points based on their y value
# Add colorbar

# 2. Create scatter plot showing:
# - x vs y (positive correlation)
# - x vs z (negative correlation)
# Side by side in subplots

# 3. Create scatter with size based on distance from origin
# size = sqrt(x^2 + y^2)
```

---

### **Exercise 3: Histograms and Distributions**

```python
# Generate different distributions
normal = np.random.randn(1000)
uniform = np.random.uniform(-3, 3, 1000)
exponential = np.random.exponential(1, 1000)

# 1. Create histogram of normal distribution
# 30 bins, with labels and title

# 2. Plot all three distributions on same plot
# Use alpha=0.5 for transparency
# Different colors, add legend

# 3. Create 3 subplots (1 row, 3 cols)
# Each showing one distribution
# Add vertical line at mean of each

# 4. For normal distribution:
# - Calculate mean and std
# - Plot histogram
# - Add vertical lines at mean, mean±std, mean±2std
# - Use different colors for each line
```

---

### **Exercise 4: Bar Charts**

```python
# Model comparison data
models = ['Model A', 'Model B', 'Model C', 'Model D']
accuracy = [0.85, 0.89, 0.87, 0.91]
training_time = [10, 25, 15, 30]  # minutes

# 1. Create bar chart of accuracy
# Color bars based on value (green if > 0.88)

# 2. Create horizontal bar chart of training time
# Sorted from fastest to slowest

# 3. Create grouped bar chart showing both metrics
# (you'll need to position bars carefully)

# 4. Create subplot with 2 charts:
# Left: accuracy bars
# Right: training time bars
```

---

### **Exercise 5: Seaborn Basics**

```python
import seaborn as sns
import pandas as pd

# Create dataset
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.randn(300),
    'feature2': np.random.randn(300),
    'feature3': np.random.randn(300),
    'category': np.random.choice(['A', 'B', 'C'], 300)
})
df['feature2'] = df['feature1'] * 2 + df['feature2']

# 1. Create histogram with KDE for feature1

# 2. Create box plots for all features grouped by category
# (hint: melt the dataframe first)

# 3. Create violin plots for feature1 by category

# 4. Create pair plot for all features
# Color by category
```

---

### **Exercise 6: Correlation Analysis**

```python
# Create correlated dataset
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'A': np.random.randn(n),
})
df['B'] = df['A'] * 2 + np.random.randn(n) * 0.3
df['C'] = -df['A'] + np.random.randn(n) * 0.5
df['D'] = np.random.randn(n)  # Uncorrelated
df['E'] = df['B'] + df['C'] + np.random.randn(n) * 0.2

# 1. Calculate correlation matrix

# 2. Create heatmap of correlations
# Annotate with values
# Use appropriate colormap (diverging around 0)

# 3. Create pair plot (scatter matrix)
# What patterns do you see?

# 4. For each pair with |correlation| > 0.7:
# Create individual scatter plot with regression line
```

---

### **Exercise 7: Real Dataset - Iris**

```python
# Load iris dataset (famous ML dataset)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 1. Create histogram for each feature
# 4 subplots in 2x2 grid

# 2. Create box plots comparing each feature across species

# 3. Create pair plot colored by species
# What separates the species?

# 4. Create scatter plot of:
# x = sepal length, y = sepal width
# Color by species, size by petal length

# 5. Create correlation heatmap for each species separately
```

---

### **Exercise 8: Time Series Visualization**

```python
# Create time series data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
value = np.cumsum(np.random.randn(365)) + 100
df = pd.DataFrame({'date': dates, 'value': value})

# 1. Create line plot of entire time series

# 2. Add 7-day rolling mean
# Plot both original and smoothed on same chart

# 3. Create subplot with 4 quarters
# Each showing one quarter of the year

# 4. Create histogram of daily changes
# (value today - value yesterday)

# 5. Create month-over-month comparison
# Box plot with each box being a month
```

---

### **Exercise 9: Model Evaluation Visualization**

```python
# Simulate model results
np.random.seed(42)
n = 100
y_true = np.random.rand(n)
y_pred = y_true + np.random.randn(n) * 0.1
y_pred = np.clip(y_pred, 0, 1)  # Keep in [0, 1]

# 1. Scatter plot: predictions vs actual
# Add diagonal line (perfect predictions)
# Color points by error magnitude

# 2. Plot residuals (y_true - y_pred) vs predictions
# Add horizontal line at 0
# Are residuals randomly distributed?

# 3. Create histogram of residuals
# Are they normally distributed?

# 4. Create learning curves
epochs = range(1, 51)
train_error = np.exp(-np.linspace(0, 2, 50)) + np.random.randn(50)*0.02
val_error = np.exp(-np.linspace(0, 1.8, 50)) + np.random.randn(50)*0.03
# Plot both on same chart
# When does overfitting start?
```

---

### **Exercise 10: Comprehensive EDA**

```python
# Create realistic dataset
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.lognormal(10.5, 0.5, n),
    'credit_score': np.random.randint(300, 850, n),
    'loan_amount': np.random.uniform(1000, 50000, n),
    'employment_years': np.random.randint(0, 40, n),
    'default': np.random.choice([0, 1], n, p=[0.8, 0.2])
})

# Add some relationships
df.loc[df['credit_score'] < 600, 'default'] = np.random.choice([0, 1], 
    sum(df['credit_score'] < 600), p=[0.4, 0.6])

# Perform COMPLETE exploratory data analysis:

# 1. Distribution of each feature (histograms)

# 2. Box plots to spot outliers

# 3. Correlation matrix heatmap

# 4. Default rate by credit score bins
# (divide credit score into ranges)

# 5. Age vs Income scatter, colored by default status

# 6. Compare distributions of defaulters vs non-defaulters
# for each feature

# 7. Create summary visualization:
# - Top: Distribution of target (default)
# - Middle row: 3 most important features
# - Bottom: Correlation heatmap

# Save as single figure with subplots
```

---

## **VISUALIZATION BEST PRACTICES**

### **Do's:**
✅ Always label axes and add titles
✅ Use color meaningfully (not just decoration)
✅ Choose appropriate plot type for data
✅ Make text readable (font size)
✅ Use legends when multiple series
✅ Keep it simple - don't over-complicate

### **Don'ts:**
❌ Don't use 3D when 2D works
❌ Don't use too many colors
❌ Don't use pie charts for >5 categories
❌ Don't start y-axis at arbitrary values
❌ Don't make plots too small or crowded
❌ Don't forget to show scale/units

---

## **PLOT TYPE SELECTION GUIDE**

| Data Type | Use This |
|-----------|----------|
| Single continuous variable | Histogram, KDE |
| Two continuous variables | Scatter plot |
| Continuous over time | Line plot |
| Categories vs continuous | Bar plot, Box plot |
| Distributions across groups | Box plot, Violin plot |
| Correlations | Heatmap, Pair plot |
| Part-of-whole | Pie chart (rarely), Stacked bar |
| Model performance | Line plot (loss/accuracy) |
| Predictions vs actual | Scatter with diagonal |

---

## **COMMON COLORMAPS**

```python
# Sequential (for continuous data)
'viridis', 'plasma', 'Blues', 'Reds'

# Diverging (for data with meaningful center)
'coolwarm', 'RdBu', 'seismic'

# Qualitative (for categories)
'Set1', 'Set2', 'tab10', 'Paired'

# Usage
plt.imshow(data, cmap='viridis')
sns.heatmap(data, cmap='coolwarm')
```

---

## **SAVING FIGURES**

```python
# Save current figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Different formats
plt.savefig('plot.pdf')
plt.savefig('plot.svg')

# With seaborn
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='x', y='y', ax=ax)
fig.savefig('seaborn_plot.png', dpi=300, bbox_inches='tight')
```

---

## **IMPORTANT NOTES**

✅ Install: `pip install matplotlib seaborn --break-system-packages`
✅ Import convention:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```
✅ Always explore data visually before modeling
✅ Use visualization to debug model issues
✅ Practice with real datasets

**Key skill**: Choose the right plot for the story you want to tell with data!
