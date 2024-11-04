import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\aswin\OneDrive\Desktop\salary_data.csv")
print(df.head())

# Extract values
x = df['YearsExperience'].values
y = df['Salary'].values

# Calculate means
X_mean = np.mean(x)
Y_mean = np.mean(y)

# Calculate slope (beta_1) and intercept (beta_0)
beta_1 = np.sum((y - Y_mean) * (x - X_mean)) / np.sum((x - X_mean) ** 2)
beta_0 = Y_mean - (beta_1 * X_mean)

# Compute predictions based on the regression line
predictions = beta_0 + beta_1 * x

print("Slope (beta_1):", beta_1)
print("Intercept (beta_0):", beta_0)


# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, predictions, color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()
