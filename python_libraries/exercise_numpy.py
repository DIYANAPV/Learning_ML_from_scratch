import numpy as np

# 1. Create array of 0 to 99
arr1 = np.arange(100)

print("Array of 0 to 99:")
print(arr1)


arr2 = np.ones((5, 5)) * 7

print("Array of 7s:")
print(arr2)

# 3. Create 3x3 identity matrix
arr3 = np.eye(3)
print("3x3 Identity Matrix:")
print(arr3)


# 4. Create array [2, 4, 6, 8, 10, ..., 100]
arr4 = np.arange(2, 102, 2)
print("Array of even numbers from 2 to 100:")
print(arr4)

# 5. Create 10 evenly spaced numbers between 0 and 1
arr5 = np.linspace(0, 1, 10)
print("10 evenly spaced numbers between 0 and 1:")
print(arr5)

# 6. Create 4x4 random matrix with values between 0 and 1
arr6 = np.random.rand(4, 4)
print("4x4 Random Matrix:")
print(arr6)


### **Exercise 2: Vectorization Challenge**

import numpy as np

def vector_add_fast(v1, v2):
    return np.add(v1, v2)

def vector_subtract_fast(v1, v2):
    return np.subtract(v1, v2)

def scalar_multiply_fast(scalar, v):
    return np.multiply(scalar, v)

def dot_product_fast(v1, v2):
    return np.dot(v1, v2)

def magnitude_fast(v):
    return np.linalg.norm(v)

# Test
a = np.array([2, 3, 1])
b = np.array([1, -1, 2])

print("Vector Addition (Fast):", vector_add_fast(a, b))
print("Vector Subtraction (Fast):", vector_subtract_fast(a, b))
print("Scalar Multiplication (Fast):", scalar_multiply_fast(2, a))
print("Dot Product (Fast):", dot_product_fast(a, b))
print("Magnitude (Fast):", magnitude_fast(a))


### **Exercise 3: Boolean Indexing**

import numpy as np

data = np.random.randint(0, 51, size=100)

# Tasks:
# 1. Find all values greater than 25
high_values = data[data > 25]

# 2. Find all even numbers
evens = data[data % 2 == 0]

# 3. Find all values between 10 and 30 (inclusive)
middle_values = data[(data >= 10) & (data <= 30)]

# 4. Replace all values > 40 with 40 (clipping)
# Don't create new array, modify in place
data[data > 40] = 40

# 5. Count how many values are divisible by 5
count = np.sum(data % 5 == 0)


### **Exercise 4: Broadcasting Practice**

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# 1. Add 100 to all elements
result1 = matrix + 100

# 2. Multiply each row by [1, 2, 3]
row_multiplier = np.array([1, 2, 3])
result2 = matrix * row_multiplier

# 3. Subtract [1, 2, 3, 4] from each column
col_subtractor = np.array([[1], [2], [3], [4]])
result3 = matrix - col_subtractor

# 4. Normalize each row (subtract row mean, divide by row std)
# Hint: Use axis parameter
normalized = (matrix - matrix.mean(axis=1, keepdims=True)) / matrix.std(axis=1, keepdims=True)

### **Exercise 5: Matrix Operations**

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# 1. Element-wise multiplication
elem_mult = A * B

# 2. Matrix multiplication (dot product)
mat_mult = np.dot(A, B)

# 3. Transpose of A
A_transpose = A.T

# 4. Determinant of A
det_A = np.linalg.det(A)

# 5. Inverse of A (if it exists)
inv_A = np.linalg.inv(A) if det_A != 0 else None

# 6. Verify A @ inv_A gives identity
verification = np.dot(A, inv_A) if inv_A is not None else None


### **Exercise 6: Reshaping Challenge**

# Create array [0, 1, 2, ..., 23]
arr = np.arange(24)

# Reshape to:
# 1. 4x6 matrix
shape1 = arr.reshape(4, 6)

# 2. 2x3x4 tensor (3D array)
shape2 = arr.reshape(2, 3, 4)

# 3. 6x4 matrix
shape3 = arr.reshape(6, 4)


# 4. Back to 1D
shape4 = arr.reshape(-1)

# 5. Column vector (24, 1)
shape5 = arr.reshape(-1, 1)


### **Exercise 7: Statistical Operations**


# Create 5x4 random matrix
data = np.random.randn(5, 4)  # Random normal distribution

# Calculate:
# 1. Mean of entire matrix
overall_mean = np.mean(data)

# 2. Mean of each column
col_mean = np.mean(data, axis=0)

# 3. Mean of each row
row_mean = np.mean(data, axis=1)

# 4. Find the maximum value in each column
column_maxs = np.max(data, axis=0)

# 5. Find index of minimum value in entire matrix
min_index = np.argmin(data)

# 6. Normalize entire matrix (mean=0, std=1)
normalized = (data - overall_mean) / np.std(data)


### **Exercise 8: Real ML Application - Data Normalization**


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
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Apply normalization
normalized_data = standardize(data)


### **Exercise 10: Image Processing Basics**


# Create fake "image" - 100x100 pixels, grayscale
image = np.random.rand(100, 100) * 255  # Values 0-255

# Tasks:
# 1. Get image dimensions
height, width = image.shape

# 2. Clip all values to range [0, 200]
clipped = np.clip(image, 0, 200)

# 3. Increase brightness by 50 (but don't exceed 255)
brightened = np.clip(image + 50, 0, 255)

# 4. Flip image vertically
flipped_vertical = np.flipud(image)

# 6. Crop center 50x50 region
cropped = image[25:75, 25:75]

# 7. Downsample to 50x50 (take every other pixel)
downsampled = image[::2, ::2]