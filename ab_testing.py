#!/usr/bin python3

import random
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Dummy Data Setup
# -------------------------------
# Suppose we have 4 variants with the following (unknown to the algorithm) true conversion rates:
true_conversion_rates = {
    'Variant A': 0.05,  # 5% conversion rate
    'Variant B': 0.08,  # 8% conversion rate
    'Variant C': 0.12,  # 12% conversion rate (best variant)
    'Variant D': 0.03   # 3% conversion rate
}

variants = list(true_conversion_rates.keys())
n_arms = len(variants)

# -------------------------------
# Parameters for the Simulation
# -------------------------------
n_rounds = 10000  # Total number of simulated user visits (trials)

# For each arm, maintain counters for successes (conversions) and failures (no conversion)
successes = {variant: 0 for variant in variants}
failures  = {variant: 0 for variant in variants}

# Lists to keep track of our choices and rewards over time
chosen_variants = []
cumulative_rewards = []  # Cumulative number of conversions over time
total_rewards = 0

# -------------------------------
# Thompson Sampling Algorithm
# -------------------------------
# For each simulated trial, sample a conversion rate for each arm from its Beta distribution,
# choose the arm with the highest sampled conversion rate, simulate a user interaction,
# and update the corresponding success or failure count.
for round in range(n_rounds):
    # Sample a conversion rate (theta) for each variant using the Beta distribution.
    # We add 1 to both the successes and failures (Beta(1,1) is a uniform prior).
    sampled_thetas = {
        variant: np.random.beta(successes[variant] + 1, failures[variant] + 1)
        for variant in variants
    }
    
    # Select the variant with the highest sampled conversion rate.
    chosen_variant = max(sampled_thetas, key=sampled_thetas.get)
    chosen_variants.append(chosen_variant)
    
    # Simulate a user interaction:
    # The user converts (reward = 1) with probability equal to the true conversion rate of the variant.
    if random.random() < true_conversion_rates[chosen_variant]:
        reward = 1
        successes[chosen_variant] += 1  # Update successes for the chosen variant
    else:
        reward = 0
        failures[chosen_variant] += 1   # Update failures for the chosen variant
    
    total_rewards += reward
    cumulative_rewards.append(total_rewards)

# -------------------------------
# Results Summary
# -------------------------------
print("True Conversion Rates:")
for variant, rate in true_conversion_rates.items():
    print(f"  {variant}: {rate:.2%}")

print("\nEstimated Successes and Failures:")
for variant in variants:
    print(f"  {variant}: {successes[variant]} successes, {failures[variant]} failures")

print("\nNumber of times each variant was selected:")
for variant in variants:
    count = chosen_variants.count(variant)
    print(f"  {variant}: {count} times")

print(f"\nTotal Conversions: {total_rewards}")
print(f"Overall Conversion Rate: {total_rewards/n_rounds:.2%}")

# -------------------------------
# Plotting the Cumulative Reward Over Time
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(cumulative_rewards)
plt.xlabel('Rounds')
plt.ylabel('Cumulative Conversions')
plt.title('Cumulative Conversions over Time using Thompson Sampling')
plt.grid(True)
plt.show()
