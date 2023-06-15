import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
n = 50
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.3 * np.random.randn(n)

# Split data into training and testing sets
n_train = 20
x_train, y_train = x[:n_train], y[:n_train]
x_test, y_test = x[n_train:], y[n_train:]

# Define model capacities to test
capacities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialize arrays to store error values
train_errors = np.zeros(len(capacities))
test_errors = np.zeros(len(capacities))

# Loop over model capacities and fit models
for i, c in enumerate(capacities):
    # Define and fit model
    model = np.poly1d(np.polyfit(x_train, y_train, c))
    
    # Calculate errors on training and testing sets
    train_errors[i] = np.mean((model(x_train) - y_train) ** 2)
    test_errors[i] = np.mean((model(x_test) - y_test) ** 2)
    
# Plot results
plt.plot(capacities, train_errors, label='Training Error')
plt.plot(capacities, test_errors, label='Generalization Error')
plt.xlabel('Model Capacity')
plt.ylabel('Mean Squared Error')
plt.title('Training Error vs Generalization Error')
plt.legend()
plt.show()