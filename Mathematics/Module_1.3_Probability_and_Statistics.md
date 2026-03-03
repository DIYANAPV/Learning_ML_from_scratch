# Module 1.3: Probability & Statistics for ML

## **Why Probability & Statistics Matter**

**Machine learning is fundamentally about uncertainty and predictions.**

- LLMs generate text by **sampling from probability distributions**
- Classification models output **probabilities** (e.g., 80% spam, 20% not spam)
- Training data is a **sample** from a larger population
- Bayesian methods update beliefs based on **evidence**
- Generative AI models learn **probability distributions** of data

If you don't understand probability, you can't understand how AI makes decisions under uncertainty.

---

## **CONCEPT 1: Probability Basics**

### **What is Probability?**

Probability measures **how likely** an event is to occur.

**Scale:** 0 to 1 (or 0% to 100%)
- `P(event) = 0` → impossible
- `P(event) = 1` → certain
- `P(event) = 0.5` → 50-50 chance

**Basic Rules:**
```
P(A or B) = P(A) + P(B) - P(A and B)
P(not A) = 1 - P(A)
P(all possible outcomes) = 1
```

---

### **Example: Coin Flip**

```
P(Heads) = 0.5
P(Tails) = 0.5
P(Heads) + P(Tails) = 1
```

### **Example: Dice Roll**

```
P(rolling 4) = 1/6 ≈ 0.167
P(rolling even) = P(2 or 4 or 6) = 3/6 = 0.5
P(not rolling 6) = 1 - 1/6 = 5/6
```

---

## **CONCEPT 2: Conditional Probability**

### **What is Conditional Probability?**

The probability of event A **given that** event B has occurred.

**Notation:** `P(A|B)` (read: "probability of A given B")

**Formula:**
```
P(A|B) = P(A and B) / P(B)
```

**Intuition:** When you know B happened, you're working with a smaller sample space.

---

### **Example: Medical Test**

```
P(Disease) = 0.01 (1% of people have it)
P(Positive Test | Disease) = 0.95 (95% sensitivity)
P(Positive Test | No Disease) = 0.05 (5% false positive)

Question: If you test positive, what's the probability you have the disease?
This is P(Disease | Positive Test) ← This is what we need Bayes' theorem for!
```

---

## **CONCEPT 3: Bayes' Theorem**

### **The Most Important Formula in ML**

Bayes' theorem lets you **update beliefs based on new evidence**.

**Formula:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**In words:**
```
P(Hypothesis|Evidence) = P(Evidence|Hypothesis) × P(Hypothesis) / P(Evidence)
```

**Components:**
- `P(A|B)` = **Posterior** (what we want to know)
- `P(B|A)` = **Likelihood** (how likely is evidence given hypothesis)
- `P(A)` = **Prior** (what we believed before seeing evidence)
- `P(B)` = **Marginal** (total probability of evidence)

---

### **Medical Test Example (Solved)**

```
Given:
P(Disease) = 0.01
P(Pos|Disease) = 0.95
P(Pos|No Disease) = 0.05

Find: P(Disease|Pos)

Step 1: Calculate P(Pos)
P(Pos) = P(Pos|Disease)×P(Disease) + P(Pos|No Disease)×P(No Disease)
       = 0.95×0.01 + 0.05×0.99
       = 0.0095 + 0.0495
       = 0.059

Step 2: Apply Bayes' Theorem
P(Disease|Pos) = P(Pos|Disease) × P(Disease) / P(Pos)
                = 0.95 × 0.01 / 0.059
                = 0.0095 / 0.059
                ≈ 0.161 or 16.1%
```

**Surprising result:** Even with positive test, only 16% chance of disease!

**Why it matters in ML:**
- Spam filters use Bayes' theorem
- Bayesian neural networks quantify uncertainty
- Prior knowledge + data = better predictions

---

## **CONCEPT 4: Probability Distributions**

### **What is a Distribution?**

A distribution describes **how probability is spread across possible values**.

---

### **1. Bernoulli Distribution**

**Binary outcome** (success/failure, yes/no, 1/0)

```
P(X=1) = p
P(X=0) = 1-p

Example: Coin flip with p=0.5
Example: Click/No-click on ad with p=0.02
```

**Why it matters:** Binary classification outputs

---

### **2. Categorical Distribution**

**Multiple discrete outcomes**

```
Example: Dice roll
P(1) = P(2) = P(3) = P(4) = P(5) = P(6) = 1/6

Example: Next word in LLM
P(word1) = 0.3
P(word2) = 0.25
P(word3) = 0.2
...
```

**Why it matters:** LLMs output probability distribution over vocabulary

---

### **3. Normal (Gaussian) Distribution**

**Bell curve** - most common in nature and ML

```
Properties:
- Symmetric around mean (μ)
- Spread controlled by standard deviation (σ)
- 68% of data within 1σ of mean
- 95% of data within 2σ of mean
- 99.7% of data within 3σ of mean
```

**Formula (don't memorize, just know it exists):**
```
P(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
```

**Example:**
```
Heights: μ=170cm, σ=10cm
IQ scores: μ=100, σ=15
```

**Why it matters:**
- Many real-world features are normally distributed
- Error terms in models often assumed normal
- Weight initialization in neural networks

---

## **CONCEPT 5: Expected Value & Variance**

### **Expected Value (Mean)**

The **average** outcome if you repeat an experiment many times.

**Formula:**
```
E[X] = Σ(x × P(x))
```

**Example - Dice Roll:**
```
E[X] = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
     = (1+2+3+4+5+6)/6
     = 21/6 = 3.5
```

---

### **Variance & Standard Deviation**

**Variance** measures how **spread out** the values are.

**Formula:**
```
Var(X) = E[(X - μ)²]
Standard Deviation (σ) = √Var(X)
```

**Intuition:**
- Low variance → values clustered near mean
- High variance → values spread out

**Why it matters:**
- Understand data spread
- Normalization techniques
- Uncertainty quantification

---

## **CONCEPT 6: Maximum Likelihood Estimation (MLE)**

### **What is MLE?**

Finding parameters that **make your observed data most likely**.

**Example:**
```
You flip a coin 10 times: H H T H H H T H H T

What's the best estimate for P(Heads)?

Answer: p = (# of heads) / (# of flips) = 7/10 = 0.7
```

This is MLE! You found the parameter (p) that maximizes the likelihood of seeing that data.

**Why it matters:**
- Core principle of training ML models
- Loss functions often derived from MLE
- Understanding why we minimize squared error, cross-entropy, etc.

---

## **YOUR EXERCISES**

### **Exercise 1: Basic Probability**

Calculate by hand:

1. You roll two dice. What's P(sum = 7)?
2. You draw a card from standard deck. What's P(Ace or King)?
3. You flip a coin 3 times. What's P(at least 2 heads)?
4. From bag with 5 red, 3 blue, 2 green balls, what's P(red)?

---

### **Exercise 2: Conditional Probability**

Given:
- 60% of emails are spam
- 80% of spam emails contain word "free"
- 10% of non-spam emails contain word "free"

Calculate:
1. P(spam and contains "free")
2. P(contains "free")
3. P(spam | contains "free") ← Use Bayes' theorem

---

### **Exercise 3: Bayes' Theorem - Spam Filter**

Build a simple spam classifier:

Given word frequencies:
```
P(spam) = 0.4
P("lottery"|spam) = 0.8
P("lottery"|not spam) = 0.1
```

Calculate: If email contains "lottery", what's P(spam|"lottery")?

Then implement in code:

```python
def bayes_spam_filter(p_spam, p_word_given_spam, p_word_given_not_spam):
    """
    Calculate P(spam|word) using Bayes' theorem
    
    Returns: probability email is spam given it contains the word
    """
    pass

# Test
result = bayes_spam_filter(0.4, 0.8, 0.1)
print(f"P(spam|word) = {result:.3f}")
```

---

### **Exercise 4: Expected Value**

Calculate by hand:

1. Expected value of rolling a 6-sided die
2. Expected value of this game:
   - Pay $1 to play
   - Roll die: if 6, win $5; otherwise win nothing
   - Should you play? (Expected profit?)

3. A lottery:
   - Ticket costs $2
   - P(win $100) = 0.01
   - P(win $10) = 0.05
   - P(win $0) = 0.94
   - Expected value?

---

### **Exercise 5: Variance Calculation**

For dataset: [2, 4, 4, 4, 5, 5, 7, 9]

Calculate by hand:
1. Mean (μ)
2. Variance
3. Standard deviation

Then verify with code:

```python
def calculate_mean(data):
    pass

def calculate_variance(data):
    # variance = average of squared differences from mean
    pass

def calculate_std(data):
    # std = sqrt(variance)
    pass
```

---

### **Exercise 6: Probability Distributions Code**

Implement these distributions:

```python
def bernoulli_sample(p, n_samples=1000):
    """
    Generate n_samples from Bernoulli(p)
    Return list of 0s and 1s
    Use: import random; random.random()
    """
    pass

def categorical_sample(probabilities, n_samples=1000):
    """
    Sample from categorical distribution
    probabilities: list that sums to 1
    Example: [0.2, 0.3, 0.5] for 3 categories
    """
    pass

def estimate_probability(samples):
    """
    Given samples, estimate probability of each outcome
    Return dictionary: {outcome: probability}
    """
    pass

# Test
samples = bernoulli_sample(0.7, 10000)
print(f"Estimated p: {sum(samples)/len(samples):.3f}")  # Should be ~0.7
```

---

### **Exercise 7: Simple Naive Bayes Classifier**

Implement a text classifier using Bayes' theorem:

```python
class NaiveBayesClassifier:
    def __init__(self):
        self.word_counts = {}  # {class: {word: count}}
        self.class_counts = {}  # {class: total_count}
        
    def train(self, texts, labels):
        """
        texts: list of text strings
        labels: list of class labels (e.g., 'spam' or 'not_spam')
        """
        pass
    
    def predict(self, text):
        """
        Return probability of each class given the text
        Use Bayes' theorem and word frequencies
        """
        pass

# Test data
train_texts = [
    "win free lottery money",
    "meeting tomorrow at 3pm",
    "free prize winner congrats",
    "project deadline reminder"
]
train_labels = ["spam", "not_spam", "spam", "not_spam"]

classifier = NaiveBayesClassifier()
classifier.train(train_texts, train_labels)

test_text = "free money winner"
result = classifier.predict(test_text)
print(f"Probabilities: {result}")
```

---

### **Exercise 8: Normal Distribution Exploration**

```python
import random
import math

def generate_normal_samples(mean, std, n_samples=1000):
    """
    Generate samples from normal distribution
    Use Box-Muller transform or any method you can find
    """
    pass

def check_empirical_rule(samples, mean, std):
    """
    Verify 68-95-99.7 rule:
    - Count % within 1 std
    - Count % within 2 std  
    - Count % within 3 std
    """
    pass

# Test
samples = generate_normal_samples(mean=0, std=1, n_samples=10000)
check_empirical_rule(samples, 0, 1)
```

---

### **Exercise 9: Maximum Likelihood Estimation**

```python
def mle_bernoulli(data):
    """
    Given binary data (0s and 1s), estimate p
    data: list of 0s and 1s
    return: estimated p
    """
    pass

def mle_normal(data):
    """
    Given data, estimate mean and variance
    return: (estimated_mean, estimated_variance)
    """
    pass

# Test
coin_flips = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]  # 7 heads out of 10
p_estimate = mle_bernoulli(coin_flips)
print(f"Estimated p: {p_estimate}")  # Should be 0.7

heights = [165, 170, 168, 172, 169, 171, 167, 173, 170, 168]
mean_est, var_est = mle_normal(heights)
print(f"Estimated mean: {mean_est:.2f}, variance: {var_est:.2f}")
```

---

## **REAL-WORLD CONNECTION**

### **How This Relates to LLMs**

1. **Next Token Prediction**: LLM outputs probability distribution over all possible next tokens
   ```
   P("the"|context) = 0.3
   P("a"|context) = 0.25
   P("is"|context) = 0.15
   ...
   ```

2. **Sampling Strategies**: Temperature controls the distribution shape
   - High temp → flatter distribution (more random)
   - Low temp → peaky distribution (more deterministic)

3. **Bayesian Updating**: As LLM reads more context, it updates probabilities

4. **Uncertainty**: Better models know when they're uncertain (high entropy in output distribution)

---

## **IMPORTANT NOTES**

✅ Do hand calculations first
✅ Implement all code yourself
✅ Test with different inputs
✅ Understand WHY formulas work
✅ Connect to real ML applications

**Key Insight**: ML is probability in disguise. Every prediction is a probability distribution!
