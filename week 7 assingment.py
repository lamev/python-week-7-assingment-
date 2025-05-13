# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ------------------------------
# Task 1: Load and Explore the Dataset
# ------------------------------

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset (demonstration - no actual missing values)
df_cleaned = df.fillna(method='ffill')

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------

# Describe numerical columns
print("\nDescriptive Statistics:")
print(df_cleaned.describe())

# Group by species and calculate mean
grouped_means = df_cleaned.groupby('target').mean()
print("\nGrouped Means by Species (target):")
print(grouped_means)

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

# Set up plotting space
plt.figure(figsize=(16, 12))

# Line chart: Sepal length over index (simulated time series)
plt.subplot(2, 2, 1)
plt.plot(df_cleaned.index, df_cleaned['sepal length (cm)'], label='Sepal Length')
plt.title("Line Chart: Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()

# Bar chart: Average petal length per species
plt.subplot(2, 2, 2)
sns.barplot(x='target', y='petal length (cm)', data=df_cleaned, ci=None)
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species (target)")
plt.ylabel("Petal Length (cm)")

# Histogram: Sepal width distribution
plt.subplot(2, 2, 3)
plt.hist(df_cleaned['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")

# Scatter plot: Sepal length vs Petal length
plt.subplot(2, 2, 4)
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df_cleaned)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")

# Display all plots
plt.tight_layout()
plt.show()
