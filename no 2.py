import numpy as np
import matplotlib.pyplot as plt

#data
x1 = 1
x2 = 0
target_y1 = 0.75
target_y2 = 0.75

# asumsi data pada inisialisasi
W13 = 0.25
W14 = -0.35
W23 = 0.75
W24 = 0.15
W35 = -1
W45 = 1
theta_3 = -0.75
theta_4 = -0.25
theta_5 = 0.15

# Learning rate
learning_rate = 0.15

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2 iterasi
num_iterations = 2
errors = []

for iteration in range(num_iterations):
    # Forward propagation
    h1 = sigmoid(W13 * x1 + W14 * x2 - theta_3)
    h2 = sigmoid(W23 * x1 + W24 * x2 - theta_4)
    y1 = sigmoid(W35 * h1 - theta_5)
    y2 = sigmoid(W45 * h2 - theta_5)

    # Calculate errors
    error_y1 = target_y1 - y1
    error_y2 = target_y2 - y2
    total_error = 0.5 * (error_y1**2 + error_y2**2)  # Mean Squared Error
    errors.append(total_error)

    # Backward propagation
    error_h1 = W35 * error_y1 * h1 * (1 - h1)
    error_h2 = W45 * error_y2 * h2 * (1 - h2)

    # Update weights
    dW35 = learning_rate * error_y1 * h1
    dW45 = learning_rate * error_y2 * h2
    dW13 = learning_rate * error_h1 * x1
    dW14 = learning_rate * error_h1 * x2
    dW23 = learning_rate * error_h2 * x1
    dW24 = learning_rate * error_h2 * x2
    dtheta_3 = learning_rate * error_h1 * (-1)
    dtheta_4 = learning_rate * error_h2 * (-1)
    dtheta_5 = learning_rate * (error_y1 + error_y2) * (-1)

    W35 += dW35
    W45 += dW45
    W13 += dW13
    W14 += dW14
    W23 += dW23
    W24 += dW24
    theta_3 += dtheta_3
    theta_4 += dtheta_4
    theta_5 += dtheta_5

# Visualisasi
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), errors, marker='o', color='blue', label='Total Error (MSE)')
plt.title('Total Error Over 2 Iterations')
plt.xlabel('Iterations')
plt.ylabel('Total Error (MSE)')
plt.xticks(range(1, num_iterations + 1))  # Set x ticks to match iterations
plt.grid(False)
plt.legend()
plt.show()

# Hasil
print("\nHasil setelah 2 iterasi:")
print("Weights:")
print("W13 =", W13)
print("W14 =", W14)
print("W23 =", W23)
print("W24 =", W24)
print("W35 =", W35)
print("W45 =", W45)
print("theta_3 =", theta_3)
print("theta_4 =", theta_4)
print("theta_5 =", theta_5)