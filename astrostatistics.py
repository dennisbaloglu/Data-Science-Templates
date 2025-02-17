import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, linregress

# Simulating astronomical data
np.random.seed(42)
n_stars = 1000

# Apparent magnitudes (simulating a normal distribution)
apparent_magnitudes = np.random.normal(loc=15, scale=2, size=n_stars)

# Distances (in parsecs, simulating a log-normal distribution)
distances = np.random.lognormal(mean=3, sigma=0.5, size=n_stars)

# Absolute magnitude calculation (M = m - 5 * log10(d / 10))
absolute_magnitudes = apparent_magnitudes - 5 * np.log10(distances / 10)

# Creating a DataFrame
data = pd.DataFrame({
    'Apparent Magnitude': apparent_magnitudes,
    'Distance (pc)': distances,
    'Absolute Magnitude': absolute_magnitudes
})

# Summary statistics
print("Summary Statistics:")
print(data.describe())

# Plot histograms
plt.figure(figsize=(12, 5))
sns.histplot(data['Apparent Magnitude'], bins=30, kde=True)
plt.xlabel("Apparent Magnitude")
plt.ylabel("Count")
plt.title("Distribution of Apparent Magnitudes")
plt.show()

plt.figure(figsize=(12, 5))
sns.histplot(data['Distance (pc)'], bins=30, kde=True)
plt.xlabel("Distance (pc)")
plt.ylabel("Count")
plt.title("Distribution of Distances")
plt.xscale('log')  # Log scale for better visualization
plt.show()

# Scatter plot of Absolute Magnitude vs Distance
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Distance (pc)'], y=data['Absolute Magnitude'])
plt.xscale('log')
plt.gca().invert_yaxis()  # Brighter stars have lower magnitudes
plt.xlabel("Distance (pc)")
plt.ylabel("Absolute Magnitude")
plt.title("Absolute Magnitude vs Distance")
plt.show()

# Detecting outliers using Z-score
z_scores = (data['Absolute Magnitude'] - data['Absolute Magnitude'].mean()) / data['Absolute Magnitude'].std()
threshold = 3  # Standard threshold for outliers
outliers = data[np.abs(z_scores) > threshold]
print("\nDetected Outliers:")
print(outliers)

# Linear regression to analyze the trend
slope, intercept, r_value, p_value, std_err = linregress(
    np.log10(data['Distance (pc)']), data['Absolute Magnitude']
)
print(f"\nLinear Regression Results:\nSlope: {slope:.3f}\nIntercept: {intercept:.3f}\nR-squared: {r_value**2:.3f}")

# Regression plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=np.log10(data['Distance (pc)']), y=data['Absolute Magnitude'], label="Data")
x_vals = np.linspace(np.log10(data['Distance (pc)']).min(), np.log10(data['Distance (pc)']).max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='red', label="Best Fit Line")
plt.gca().invert_yaxis()
plt.xlabel("Log Distance (pc)")
plt.ylabel("Absolute Magnitude")
plt.title("Linear Regression of Absolute Magnitude vs Log Distance")
plt.legend()
plt.show()
