#Exercise 3: Bayes' Theorem - Spam Filter


def bayes_spam_filter(p_spam, p_word_given_spam, p_word_given_not_spam):
    p_word = p_word_given_spam * p_spam + p_word_given_not_spam * (1 - p_spam)
    return (p_word_given_spam * p_spam) / p_word if p_word else 0

result = bayes_spam_filter(0.4, 0.8, 0.1)
print(f"P(spam|word) = {result:.3f}")


#Exercise 5: Variance Calculation

def calculate_mean(data):
    return sum(data) / len(data) if data else 0

def calculate_variance(data):
    mean = calculate_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data) if data else 0

def calculate_std(data):
    variance = calculate_variance(data)
    return variance ** 0.5


#Exercise 6: Probability Distributions Code

def bernoulli_sample(p, n_samples=1000):
    """
    Generate n_samples from Bernoulli(p)
    Return list of 0s and 1s
    Use: import random; random.random()
    """
    import random
    return [1 if random.random() < p else 0 for _ in range(n_samples)]
