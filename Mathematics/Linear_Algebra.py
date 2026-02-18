#Exercise 1: Basic Vector Operations

a = [2, 3, 1]
b = [1, -1, 2]

a + b = [2 + 1, 3 + (-1), 1 + 2] = [3, 2, 3]
a - b = [2 - 1, 3 - (-1), 1 - 2] = [1, 4, -1]
3*a = [3*2, 3*3, 3*1] = [6, 9, 3]
a.b = (2*1) + (3*(-1)) + (1*2) = 2 - 3 + 2 = 1
a = (2^2 + 3^2 + 1^2)**0.5 = (4 + 9 + 1)**0.5 = (14)**0.5 = 3.74


#Exercise 2: Code Challenge

def vector_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def vector_subtract(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def scalar_multiply(scalar, v):
    return [scalar * x for x in v]

def dot_product(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def magnitude(v):
    return sum(x**2 for x in v) ** 0.5


#Exercise 3: Understanding Similarity

#Calculate dot products and determine relationships:

v1 = [1, 0]
v2 = [0, 1]

print(dot_product(v1, v2))  # Output: 0 (orthogonal)

