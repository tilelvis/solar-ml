
import numpy as np

# Part 1: Core Implementation (House Price Predictor)

# Data: Synthetic dataset for Size (X) vs. Price (y)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

# Initialize parameters
m = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
iterations = 50

# History storage
history = []

# Gradient Descent Training Loop
for i in range(iterations):
    # Model: Linear prediction function
    y_pred = m * X + b

    # Loss Function: Mean Squared Error (MSE)
    loss = np.mean((y_pred - y)**2)

    # Store history
    history.append((m, b, loss))

    # Derivatives (Gradients)
    # The partial derivative of the loss with respect to m is: 2 * mean((y_pred - y) * X)
    dm = 2 * np.mean((y_pred - y) * X)
    # The partial derivative of the loss with respect to b is: 2 * mean(y_pred - y)
    db = 2 * np.mean(y_pred - y)

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

print(f"Final parameters: m = {m}, b = {b}")

# Part 2: Interactive Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a meshgrid for the loss surface
m_vals = np.linspace(0, 2, 100)
b_vals = np.linspace(-1, 2, 100)
M, B = np.meshgrid(m_vals, b_vals)

# Calculate the loss for each point on the grid
loss_vals = np.zeros(M.shape)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        y_pred = M[i, j] * X + B[i, j]
        loss_vals[i, j] = np.mean((y_pred - y)**2)

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the loss surface
ax.plot_surface(M, B, loss_vals, cmap='viridis', alpha=0.6)

# Plot the descent path
history = np.array(history)
ax.plot(history[:, 0], history[:, 1], history[:, 2], 'r-o', label='Gradient Descent Path')

# Add labels and title
ax.set_xlabel('m (slope)')
ax.set_ylabel('b (intercept)')
ax.set_zlabel('Loss (MSE)')
ax.set_title('Gradient Descent Visualization')
ax.legend()

# Show the plot
plt.show()
