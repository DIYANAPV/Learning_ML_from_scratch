# Module 1.2: Calculus for ML

## **Why Calculus Matters**

**Calculus is how neural networks learn.** Every time a model trains, it's using calculus to figure out how to get better.

- **Derivatives** → How much does output change when input changes?
- **Gradient** → Which direction should we move to improve?
- **Backpropagation** → Chain rule applied to neural networks
- **Optimization** → Using gradients to minimize loss

Without calculus, you can't understand how LLMs actually learn from data.

---

## **CONCEPT 1: Derivatives**

### **What is a Derivative?**

A derivative tells you **the rate of change** or **slope** of a function at a point.

**Intuition:**
- If you're driving, derivative = your speed (how fast position changes)
- In ML, derivative = how much the loss changes when you tweak a parameter

**Notation:**
- `f(x)` is a function
- `f'(x)` or `df/dx` is its derivative

---

### **Basic Derivative Rules**

You need to memorize these:

| Function | Derivative |
|----------|-----------|
| `f(x) = c` (constant) | `f'(x) = 0` |
| `f(x) = x` | `f'(x) = 1` |
| `f(x) = x²` | `f'(x) = 2x` |
| `f(x) = x³` | `f'(x) = 3x²` |
| `f(x) = xⁿ` | `f'(x) = n·xⁿ⁻¹` |
| `f(x) = eˣ` | `f'(x) = eˣ` |
| `f(x) = ln(x)` | `f'(x) = 1/x` |

**Power Rule:** `d/dx[xⁿ] = n·xⁿ⁻¹`

**Examples:**
```
f(x) = x²     →  f'(x) = 2x
f(x) = x³     →  f'(x) = 3x²
f(x) = 5x²    →  f'(x) = 10x
f(x) = x⁴ + 3x² → f'(x) = 4x³ + 6x
```

---

### **Sum Rule**

Derivative of a sum = sum of derivatives:
```
f(x) = x² + x³
f'(x) = 2x + 3x²
```

---

## **CONCEPT 2: Partial Derivatives**

### **What are Partial Derivatives?**

When a function has **multiple inputs**, partial derivatives tell you how the output changes with respect to **one specific input**, holding others constant.

**Example:**
```
f(x, y) = x² + 3xy + y²
```

- **Partial derivative with respect to x:** Treat y as constant
  ```
  ∂f/∂x = 2x + 3y
  ```

- **Partial derivative with respect to y:** Treat x as constant
  ```
  ∂f/∂y = 3x + 2y
  ```

**Why it matters:** Neural networks have millions of parameters. We need to know how changing EACH parameter affects the loss.

---

## **CONCEPT 3: The Chain Rule**

### **What is the Chain Rule?**

When functions are **nested** (composed), the chain rule tells you how to find the derivative.

**Formula:**
```
If h(x) = f(g(x)), then h'(x) = f'(g(x)) · g'(x)
```

**Example:**
```
h(x) = (x² + 1)³

Let g(x) = x² + 1  →  g'(x) = 2x
Let f(u) = u³      →  f'(u) = 3u²

h'(x) = f'(g(x)) · g'(x)
      = 3(x² + 1)² · 2x
      = 6x(x² + 1)²
```

**Why it matters:** 
- **This IS backpropagation!** 
- Neural networks are nested functions: `output = f₃(f₂(f₁(input)))`
- Chain rule lets us compute gradients layer by layer

---

## **CONCEPT 4: Gradient**

### **What is a Gradient?**

The gradient is a **vector of all partial derivatives**.

For function `f(x, y, z)`:
```
∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
```

**Example:**
```
f(x, y) = x² + y²

∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

At point (3, 4):
∇f = [6, 8]
```

**Why it matters:**
- Gradient points in direction of **steepest increase**
- Negative gradient points toward **minimum**
- **Gradient descent** = moving opposite to gradient to minimize loss

---

## **CONCEPT 5: Gradient Descent Intuition**

### **How Neural Networks Learn**

1. Start with random weights
2. Calculate loss (how wrong the predictions are)
3. Compute gradient (which direction makes loss worse)
4. Move in **opposite direction** (to make loss better)
5. Repeat until loss is minimized

**Formula:**
```
new_weight = old_weight - learning_rate × gradient
```

**Visual Example:**
```
Loss = (prediction - actual)²

If gradient > 0: weight is too high → decrease it
If gradient < 0: weight is too low → increase it
```

---

## **YOUR EXERCISES**

### **Exercise 1: Basic Derivatives**

Calculate derivatives BY HAND:

1. `f(x) = x³`
2. `f(x) = 5x²`
3. `f(x) = x⁴ + 2x² + 1`
4. `f(x) = 3x³ - 4x + 7`
5. `f(x) = x⁵ - 2x³ + x`

For each, also calculate `f'(2)` (derivative at x=2).

---

### **Exercise 2: Partial Derivatives**

Calculate partial derivatives BY HAND:

For `f(x, y) = x² + 2xy + y²`:
1. Find `∂f/∂x`
2. Find `∂f/∂y`
3. Calculate both at point `(x=3, y=4)`

For `f(x, y, z) = x²y + yz² + 3x`:
1. Find `∂f/∂x`
2. Find `∂f/∂y`
3. Find `∂f/∂z`

---

### **Exercise 3: Chain Rule**

Calculate derivatives using chain rule BY HAND:

1. `h(x) = (x² + 1)²`
2. `h(x) = (2x + 3)³`
3. `h(x) = (x³ - x)⁴`

---

### **Exercise 4: Gradient Calculation**

For `f(x, y) = x² + y² - 2xy`:

1. Calculate `∇f` (gradient)
2. Evaluate gradient at point `(1, 1)`
3. Evaluate gradient at point `(2, 3)`

---

### **Exercise 5: Code Challenge - Gradient Descent**

Implement simple gradient descent from scratch:

```python
def gradient_descent_1d(f, df, x_start, learning_rate, num_iterations):
    """
    Minimize function f using gradient descent
    
    f: function to minimize
    df: derivative of f
    x_start: starting point
    learning_rate: step size
    num_iterations: how many steps
    
    Returns: list of x values at each iteration
    """
    pass

# Test with f(x) = x² (minimum should be at x=0)
def f(x):
    return x**2

def df(x):
    return 2*x

# Start at x=10, should converge to x=0
history = gradient_descent_1d(f, df, x_start=10, learning_rate=0.1, num_iterations=20)

# Print the journey
for i, x in enumerate(history):
    print(f"Step {i}: x = {x:.4f}, f(x) = {f(x):.4f}")
```

Expected: x should get closer and closer to 0.

---

### **Exercise 6: Understanding Learning Rate**

Using your gradient_descent_1d function:

1. Try `learning_rate = 0.01` (too small)
2. Try `learning_rate = 0.1` (good)
3. Try `learning_rate = 0.5` (what happens?)
4. Try `learning_rate = 1.1` (too large - diverges!)

Document what happens in each case and why.

---

### **Exercise 7: 2D Gradient Descent**

Implement gradient descent for 2D function:

```python
def gradient_descent_2d(f, grad_f, x_start, y_start, learning_rate, num_iterations):
    """
    Minimize function f(x,y) using gradient descent
    
    grad_f: function that returns [∂f/∂x, ∂f/∂y]
    """
    pass

# Test with f(x,y) = x² + y² (minimum at (0,0))
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return [2*x, 2*y]  # [∂f/∂x, ∂f/∂y]

# Start at (5, 5), should converge to (0, 0)
history = gradient_descent_2d(f, grad_f, x_start=5, y_start=5, 
                               learning_rate=0.1, num_iterations=50)
```

---

## **BONUS: Visualize Gradient Descent**

If you finish early, create a simple visualization:
- Plot the function
- Show the path gradient descent takes
- Mark the starting point and minimum

---

## **IMPORTANT NOTES**

✅ Do hand calculations first
✅ Write all code yourself
✅ Understand WHY each step works
✅ Test with different values
✅ Show your work before moving forward

**The goal:** Understand that gradient descent is just "follow the slope downhill" using calculus.

This is EXACTLY how neural networks learn - they're just doing this in millions of dimensions!
