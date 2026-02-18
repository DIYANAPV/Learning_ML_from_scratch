#Exercise 5: Code Challenge - Gradient Descent


def gradient_descent_1d(f, df, x_start, learning_rate, num_iterations):

    history = [x_start]

    x = x_start
    for _ in range(num_iterations):
        x -= learning_rate * df(x)
        history.append(x)
    return history


"""❓ What does _ mean in for _ in range(...)?

_ is just a variable name.

Normally:

for i in range(10):


i is the loop counter.

But if you don’t need the counter, Python programmers often use _ to mean:

"I don’t care about this value."

Example:

for _ in range(5):
    print("Hello")


We repeat 5 times but never use the index.

You could also write:

for i in range(num_iterations):


and it works the same.

✅ When to use _

Use _ when the loop variable is unused.
"""



# Start at x=10, should converge to x=0

history = gradient_descent_1d(f, df, x_start=10, learning_rate=0.1, num_iterations=20)
# Print the journey

for i, x in enumerate(history):
    print(f"Step {i}: x = {x:.4f}, f(x) = {f(x):.4f}")



#Exercise 7: 2D Gradient Descent

def gradient_descent_2d(f, grad_f, x_start, y_start, learning_rate, num_iterations):
    history = []
    x, y = x_start, y_start
    for _ in range(num_iterations):
        grad_x, grad_y = grad_f(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        history.append((x, y))
    return history

# Start at (5, 5), should converge to (0, 0)
history = gradient_descent_2d(f, grad_f, x_start=5, y_start=5, learning_rate=0.1, num_iterations=50)

# Print the journey
for i, (x, y) in enumerate(history):
    print(f"Step {i}: (x, y) = ({x:.4f}, {y:.4f}), f(x, y) = {f(x, y):.4f}")
