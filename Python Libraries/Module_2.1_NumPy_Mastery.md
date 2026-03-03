# Module 2.1: NumPy Mastery

## **Why NumPy Matters**

**NumPy is the foundation of ALL data science and ML in Python.**

- **Speed**: 10-100x faster than pure Python loops
- **Vectorization**: Write operations on entire arrays at once
- **Memory Efficient**: Stores data in contiguous blocks
- **Foundation**: PyTorch, TensorFlow, Pandas all built on NumPy concepts

Every ML library you'll use expects you to understand NumPy. It's non-negotiable.

---

## **CONCEPT 1: NumPy Arrays vs Python Lists**

### **Python Lists - Slow**

```python
# Pure Python - SLOW
a = [1, 2, 3, 4, 5]
b = [10, 20, 30, 40, 50]

# Element-wise addition requires loop
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])
# result = [11, 22, 33, 44, 55]
```

### **NumPy Arrays - Fast**

```python
import numpy as np

# NumPy - FAST
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Element-wise addition - single operation!
result = a + b
# result = array([11, 22, 33, 44, 55])
```

**Why faster?**
- NumPy uses C under the hood
- Operations vectorized (parallel processing)
- Fixed data types (no type checking per element)

---

## **CONCEPT 2: Creating Arrays**

### **From Lists**

```python
import numpy as np

# 1D array
a = np.array([1, 2, 3, 4, 5])

# 2D array (matrix)
b = np.array([[1, 2, 3],
              [4, 5, 6]])

# Check shape
print(a.shape)  # (5,)
print(b.shape)  # (2, 3) - 2 rows, 3 columns
```

### **Special Arrays**

```python
# All zeros
zeros = np.zeros((3, 4))  # 3x4 matrix of zeros

# All ones
ones = np.ones((2, 3))    # 2x3 matrix of ones

# Identity matrix
identity = np.eye(3)      # 3x3 identity matrix

# Range
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Evenly spaced
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Random
random = np.random.rand(3, 3)    # 3x3 random values [0, 1)
random_int = np.random.randint(0, 10, size=(3, 3))  # 3x3 random integers
```

---

## **CONCEPT 3: Array Operations**

### **Element-wise Operations**

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Arithmetic - all element-wise
print(a + b)    # [11, 22, 33, 44]
print(a - b)    # [-9, -18, -27, -36]
print(a * b)    # [10, 40, 90, 160]
print(a / b)    # [0.1, 0.1, 0.1, 0.1]
print(a ** 2)   # [1, 4, 9, 16]

# With scalars
print(a + 10)   # [11, 12, 13, 14]
print(a * 2)    # [2, 4, 6, 8]
```

### **Universal Functions (ufuncs)**

```python
a = np.array([1, 2, 3, 4])

# Mathematical functions
print(np.sqrt(a))      # Square root of each element
print(np.exp(a))       # e^x for each element
print(np.log(a))       # Natural log of each element
print(np.sin(a))       # Sine of each element

# Aggregations
print(np.sum(a))       # Sum all elements
print(np.mean(a))      # Average
print(np.std(a))       # Standard deviation
print(np.max(a))       # Maximum value
print(np.min(a))       # Minimum value
```

---

## **CONCEPT 4: Indexing and Slicing**

### **1D Arrays**

```python
a = np.array([10, 20, 30, 40, 50])

# Indexing (same as lists)
print(a[0])      # 10
print(a[-1])     # 50

# Slicing
print(a[1:4])    # [20, 30, 40]
print(a[:3])     # [10, 20, 30]
print(a[2:])     # [30, 40, 50]
print(a[::2])    # [10, 30, 50] - every 2nd element
```

### **2D Arrays (Matrices)**

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Access single element
print(matrix[0, 0])     # 1
print(matrix[1, 2])     # 6

# Access row
print(matrix[0])        # [1, 2, 3]
print(matrix[0, :])     # [1, 2, 3] - explicit

# Access column
print(matrix[:, 0])     # [1, 4, 7]

# Slicing
print(matrix[:2, :2])   # First 2 rows, first 2 cols
# [[1, 2],
#  [4, 5]]
```

### **Boolean Indexing** (Very Important for ML)

```python
a = np.array([1, 2, 3, 4, 5])

# Create boolean mask
mask = a > 3
print(mask)      # [False, False, False, True, True]

# Use mask to filter
print(a[mask])   # [4, 5]

# Or in one line
print(a[a > 3])  # [4, 5]

# Multiple conditions
print(a[(a > 2) & (a < 5)])  # [3, 4]
```

---

## **CONCEPT 5: Broadcasting**

### **What is Broadcasting?**

NumPy automatically "stretches" arrays to make shapes compatible.

**Rule**: Arrays are compatible if dimensions match OR one is 1.

```python
# Scalar broadcasting
a = np.array([1, 2, 3])
print(a + 5)  # [6, 7, 8]
# 5 is "broadcast" to [5, 5, 5]

# 1D + 1D
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(a + b)  # [11, 22, 33]

# 2D + 1D (row-wise)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
row = np.array([10, 20, 30])
print(matrix + row)
# [[11, 22, 33],
#  [14, 25, 36]]
# row is broadcast to each row of matrix

# 2D + 1D (column-wise)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
col = np.array([[10],
                [20]])
print(matrix + col)
# [[11, 12, 13],
#  [24, 25, 26]]
```

**Why it matters**: Efficient operations without explicit loops!

---

## **CONCEPT 6: Matrix Operations**

### **Dot Product**

```python
# Vector dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print(np.dot(A, B))
# Or use @ operator
print(A @ B)
```

### **Transpose**

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.T)
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### **Other Linear Algebra**

```python
# Determinant
A = np.array([[1, 2],
              [3, 4]])
print(np.linalg.det(A))

# Inverse
print(np.linalg.inv(A))

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

---

## **CONCEPT 7: Reshaping**

```python
a = np.array([1, 2, 3, 4, 5, 6])

# Reshape to 2x3
b = a.reshape(2, 3)
print(b)
# [[1, 2, 3],
#  [4, 5, 6]]

# Reshape to 3x2
c = a.reshape(3, 2)
print(c)
# [[1, 2],
#  [3, 4],
#  [5, 6]]

# Flatten back to 1D
print(b.flatten())  # [1, 2, 3, 4, 5, 6]

# Add dimension
d = a.reshape(-1, 1)  # Column vector
print(d.shape)  # (6, 1)
```

**Why it matters**: Neural networks need specific input shapes!

---

## **CONCEPT 8: Axis Operations**

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Sum along axis
print(np.sum(matrix))           # 21 - sum all elements
print(np.sum(matrix, axis=0))   # [5, 7, 9] - sum columns
print(np.sum(matrix, axis=1))   # [6, 15] - sum rows

# Mean along axis
print(np.mean(matrix, axis=0))  # [2.5, 3.5, 4.5] - mean of columns
print(np.mean(matrix, axis=1))  # [2, 5] - mean of rows

# Max along axis
print(np.max(matrix, axis=0))   # [4, 5, 6] - max of columns
print(np.max(matrix, axis=1))   # [3, 6] - max of rows
```

**Remember:**
- `axis=0` → operate on columns (vertically)
- `axis=1` → operate on rows (horizontally)

---

## **YOUR EXERCISES**

### **Exercise 1: Array Creation**

Create these arrays WITHOUT loops:

```python
import numpy as np

# 1. Create array of 0 to 99
arr1 = None  # Use np.arange

# 2. Create 5x5 matrix of all 7s
arr2 = None  # Use np.ones and multiplication

# 3. Create 3x3 identity matrix
arr3 = None  # Use np.eye

# 4. Create array [2, 4, 6, 8, 10, ..., 100]
arr4 = None  # Use np.arange with step

# 5. Create 10 evenly spaced numbers between 0 and 1
arr5 = None  # Use np.linspace

# 6. Create 4x4 random matrix with values between 0 and 1
arr6 = None  # Use np.random.rand

# Print all to verify
```

---

### **Exercise 2: Vectorization Challenge**

Convert your Module 1.1 vector functions to use NumPy:

```python
import numpy as np

# Before (pure Python with loops):
def vector_add_slow(v1, v2):
    result = []
    for i in range(len(v1)):
        result.append(v1[i] + v2[i])
    return result

# After (NumPy - no loops):
def vector_add_fast(v1, v2):
    # YOUR CODE HERE
    pass

def vector_subtract_fast(v1, v2):
    # YOUR CODE HERE
    pass

def scalar_multiply_fast(scalar, v):
    # YOUR CODE HERE
    pass

def dot_product_fast(v1, v2):
    # YOUR CODE HERE - use np.dot
    pass

def magnitude_fast(v):
    # YOUR CODE HERE - use np.sqrt and np.sum
    pass

# Test
a = np.array([2, 3, 1])
b = np.array([1, -1, 2])

# All should work now
```

---

### **Exercise 3: Boolean Indexing**

```python
# Create array of 100 random integers between 0 and 50
data = np.random.randint(0, 51, size=100)

# Tasks:
# 1. Find all values greater than 25
high_values = None

# 2. Find all even numbers
evens = None

# 3. Find all values between 10 and 30 (inclusive)
middle_values = None

# 4. Replace all values > 40 with 40 (clipping)
# Don't create new array, modify in place

# 5. Count how many values are divisible by 5
count = None  # Hint: use np.sum on boolean array
```

---

### **Exercise 4: Broadcasting Practice**

```python
# Matrix of shape (4, 3)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# 1. Add 100 to all elements
result1 = None

# 2. Multiply each row by [1, 2, 3]
row_multiplier = np.array([1, 2, 3])
result2 = None

# 3. Subtract [1, 2, 3, 4] from each column
col_subtractor = np.array([[1], [2], [3], [4]])
result3 = None

# 4. Normalize each row (subtract row mean, divide by row std)
# Hint: Use axis parameter
normalized = None
```

---

### **Exercise 5: Matrix Operations**

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# 1. Element-wise multiplication
elem_mult = None

# 2. Matrix multiplication (dot product)
mat_mult = None

# 3. Transpose of A
A_transpose = None

# 4. Determinant of A
det_A = None

# 5. Inverse of A (if it exists)
inv_A = None

# 6. Verify A @ inv_A gives identity
verification = None  # Should be close to [[1, 0], [0, 1]]
```

---

### **Exercise 6: Reshaping Challenge**

```python
# Create array [0, 1, 2, ..., 23]
arr = np.arange(24)

# Reshape to:
# 1. 4x6 matrix
shape1 = None

# 2. 2x3x4 tensor (3D array)
shape2 = None

# 3. 6x4 matrix
shape3 = None

# 4. Back to 1D
shape4 = None

# 5. Column vector (24, 1)
shape5 = None
```

---

### **Exercise 7: Statistical Operations**

```python
# Create 5x4 random matrix
data = np.random.randn(5, 4)  # Random normal distribution

# Calculate:
# 1. Mean of entire matrix
overall_mean = None

# 2. Mean of each column
column_means = None  # Should have 4 values

# 3. Standard deviation of each row
row_stds = None  # Should have 5 values

# 4. Find the maximum value in each column
column_maxs = None

# 5. Find index of minimum value in entire matrix
min_index = None  # Use np.argmin

# 6. Normalize entire matrix (mean=0, std=1)
normalized = None  # (data - mean) / std
```

---

### **Exercise 8: Real ML Application - Data Normalization**

```python
# Simulate dataset: 100 samples, 5 features
# Features have different scales
np.random.seed(42)
data = np.random.randn(100, 5) * np.array([1, 10, 100, 1000, 10000])

print("Original data shape:", data.shape)
print("Feature means:", np.mean(data, axis=0))
print("Feature stds:", np.std(data, axis=0))

# Task: Normalize each feature to mean=0, std=1
# This is called "standardization" or "z-score normalization"

def standardize(X):
    """
    Standardize features by removing mean and scaling to unit variance
    X: shape (n_samples, n_features)
    Returns: standardized X
    """
    # YOUR CODE HERE
    # Hint: Use broadcasting to subtract means and divide by stds
    pass

# Apply normalization
normalized_data = standardize(data)

# Verify
print("\nNormalized data shape:", normalized_data.shape)
print("Feature means (should be ~0):", np.mean(normalized_data, axis=0))
print("Feature stds (should be ~1):", np.std(normalized_data, axis=0))
```

---

### **Exercise 9: Vectorization Speed Test**

Compare pure Python vs NumPy:

```python
import time
import numpy as np

# Create large arrays
size = 1000000
a_list = list(range(size))
b_list = list(range(size))
a_numpy = np.arange(size)
b_numpy = np.arange(size)

# Pure Python version
start = time.time()
result_python = [a_list[i] + b_list[i] for i in range(size)]
python_time = time.time() - start

# NumPy version
start = time.time()
result_numpy = a_numpy + b_numpy
numpy_time = time.time() - start

print(f"Python time: {python_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")

# Now do the same for:
# 1. Element-wise multiplication
# 2. Computing squares
# 3. Finding mean
```

---

### **Exercise 10: Image Processing Basics**

Images are just NumPy arrays!

```python
# Create fake "image" - 100x100 pixels, grayscale
image = np.random.rand(100, 100) * 255  # Values 0-255

# Tasks:
# 1. Get image dimensions
height, width = None, None

# 2. Clip all values to range [0, 200]
clipped = None  # Use np.clip

# 3. Increase brightness by 50 (but don't exceed 255)
brightened = None

# 4. Flip image vertically
flipped_vertical = None  # Hint: use slicing [::-1]

# 5. Flip image horizontally
flipped_horizontal = None

# 6. Crop center 50x50 region
cropped = None

# 7. Downsample to 50x50 (take every other pixel)
downsampled = None  # Use slicing with step
```

---

## **PERFORMANCE TIPS**

### **Do's:**
✅ Use vectorized operations instead of loops
✅ Use broadcasting when possible
✅ Preallocate arrays when size is known
✅ Use views instead of copies when possible

### **Don'ts:**
❌ Don't use Python loops on NumPy arrays
❌ Don't convert to lists unless necessary
❌ Don't use `.append()` in loops (very slow)
❌ Don't create unnecessary copies

---

## **IMPORTANT NOTES**

✅ Install NumPy: `pip install numpy --break-system-packages`
✅ Import convention: `import numpy as np` (always!)
✅ Master broadcasting - it's powerful and confusing at first
✅ axis=0 is columns, axis=1 is rows
✅ Practice until vectorization becomes natural

**This is essential**: Everything in ML uses these concepts. PyTorch tensors work almost identically to NumPy arrays!
