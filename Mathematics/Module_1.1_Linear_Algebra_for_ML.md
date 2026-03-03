# Module 1.1: Linear Algebra for ML

## **Why Linear Algebra Matters**

Before we dive in, understand this: **Linear algebra is the language of machine learning**. 

- Neural networks? Matrix multiplications.
- Transformers/LLMs? Attention is matrix operations.
- Word embeddings? Vectors in high-dimensional space.
- Training models? Gradient descent moves through vector spaces.

If you understand linear algebra, AI/ML stops being magic and becomes **math you can see and control**.

---

## **CONCEPT 1: Vectors**

### **What is a Vector?**

A vector is just a **list of numbers** that represents:
- A point in space
- A direction and magnitude
- Features of data

**Examples:**
```
[2, 3]           → 2D vector (x=2, y=3)
[1, 2, 3]        → 3D vector
[0.5, 0.3, 0.2]  → Could be word probabilities
```

In ML, everything becomes vectors:
- An image: `[pixel1, pixel2, ..., pixelN]`
- A word embedding: `[0.2, -0.5, 0.8, ..., 0.1]` (often 768 or 1024 dimensions!)
- A sentence representation in LLMs: A vector!

---

### **Vector Operations**

#### **1. Vector Addition**
Add corresponding elements:
```
[1, 2] + [3, 4] = [1+3, 2+4] = [4, 6]
```

**Why it matters:** Combining word embeddings, adding corrections in optimization.

#### **2. Scalar Multiplication**
Multiply each element by a number:
```
2 × [1, 2, 3] = [2, 4, 6]
```

**Why it matters:** Scaling features, learning rates in gradient descent.

#### **3. Dot Product** (SUPER IMPORTANT)
Multiply corresponding elements and sum:
```
[1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

**Why it matters:** 
- **Similarity measure** (how similar are two vectors?)
- Used in attention mechanisms in transformers
- Core of neural network computations

---

### **Vector Length (Magnitude)**

How "long" is a vector?
```
||[3, 4]|| = √(3² + 4²) = √(9 + 16) = √25 = 5
```

**Formula:** `||v|| = √(v₁² + v₂² + ... + vₙ²)`

**Why it matters:** Normalizing embeddings, understanding vector spaces.

---

## **YOUR FIRST EXERCISES**

I want you to do these **by hand first** (with calculator for square roots), then **write Python code** to verify.

### **Exercise 1: Basic Vector Operations**

Given:
```
a = [2, 3, 1]
b = [1, -1, 2]
```

Calculate BY HAND:
1. `a + b`
2. `a - b`
3. `3 × a`
4. `a · b` (dot product)
5. `||a||` (magnitude of a)

**Then write Python code** (use lists and loops, NO NumPy yet) to verify your answers.

---

### **Exercise 2: Code Challenge**

Write Python functions (from scratch, no NumPy):

```python
def vector_add(v1, v2):
    # Add two vectors
    pass

def vector_subtract(v1, v2):
    # Subtract v2 from v1
    pass

def scalar_multiply(scalar, v):
    # Multiply vector by scalar
    pass

def dot_product(v1, v2):
    # Calculate dot product
    pass

def magnitude(v):
    # Calculate vector magnitude
    pass
```

Test with:
```python
a = [2, 3, 1]
b = [1, -1, 2]
c = [1, 0, 0, 1]
d = [0, 1, 1, 0]
```

---

### **Exercise 3: Understanding Similarity**

The dot product measures similarity. Two vectors are:
- **Perpendicular** (orthogonal) if dot product = 0
- **Similar direction** if dot product > 0
- **Opposite direction** if dot product < 0

Calculate dot products and determine relationships:
1. `[1, 0]` and `[0, 1]`
2. `[1, 1]` and `[1, -1]`
3. `[2, 3]` and `[4, 6]`

What do you notice about vector 3?

---

## **IMPORTANT RULES**

✅ Do ALL calculations by hand first
✅ Write ALL code yourself (no AI, no copy-paste)
✅ Test your functions thoroughly
✅ Google for syntax ONLY, not for solutions
✅ Show me your work before moving forward

---

**Start with Exercise 1 - show me your hand calculations AND Python code when done!**
