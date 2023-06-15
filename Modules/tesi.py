import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softplus(x):
    return np.log(1 + np.exp(x))

def tanh(x):
    return np.tanh(x)

# Generate x values
x = np.linspace(-5, 5, 1000)

# Plot the functions in separate subplots
fig, axs = plt.subplots(1, 4, figsize=(40, 4))
axs[0].plot(x, relu(x))
axs[0].set_title('ReLU')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[ 1].plot(x, sigmoid(x))
axs[ 1].set_title('Sigmoid')
axs[ 1].set_xlabel('x')
axs[ 1].set_ylabel('y')
axs[2].plot(x, softplus(x))
axs[2].set_title('Softplus')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[3].plot(x, tanh(x))
axs[3].set_title('Hyperbolic Tangent')
axs[3].set_xlabel('x')
axs[3].set_ylabel('y')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Show the plot
plt.show()