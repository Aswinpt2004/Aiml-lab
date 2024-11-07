import math
import numpy as np

# Define the XOR dataset
data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# Initialize parameters
alpha = 0.5
np.random.seed(42)
w11, w12 = np.random.uniform(-1, 1, (2,))
w21, w22 = np.random.uniform(-1, 1, (2,))

w_out1, w_out2 = np.random.uniform(-1, 1, (2,))


def sigmoid(y):
    return 1 / (1 + math.exp(-y))


def sigmoid_derivative(f_x):
    return f_x * (1 - f_x)


# Train the network
epochs = 10000  # iterations
for epoch in range(epochs):
    total_error = 0
    for inputs, actual in data:
        x1, x2 = inputs

        y1 = x1 * w11 + x2 * w21
        f_y1 = sigmoid(y1)

        y2 = x1 * w12 + x2 * w22
        f_y2 = sigmoid(y2)

        y_out = f_y1 * w_out1 + f_y2 * w_out2
        f_y_out = sigmoid(y_out)

        # error
        error = actual - f_y_out
        total_error += error ** 2

        # Backpropagation
        fdash_y_out = sigmoid_derivative(f_y_out)

        delta_w_out1 = alpha * error * 1 * f_y1
        delta_w_out2 = alpha * error * 1 * f_y2

        w_out1 += delta_w_out1
        w_out2 += delta_w_out2

        error_y1 = error * fdash_y_out * w_out1
        error_y2 = error * fdash_y_out * w_out2

        fdash_y1 = sigmoid_derivative(f_y1)
        fdash_y2 = sigmoid_derivative(f_y2)

        delta_w11 = alpha * error_y1 * fdash_y1 * x1
        delta_w21 = alpha * error_y1 * fdash_y1 * x2
        delta_w12 = alpha * error_y2 * fdash_y2 * x1
        delta_w22 = alpha * error_y2 * fdash_y2 * x2

        w11 += delta_w11
        w21 += delta_w21
        w12 += delta_w12
        w22 += delta_w22
    total_error /= len(data)

    if epoch % 1000 == 0:
        print("Epoch ", epoch, "Total Error:", total_error)

print("\nTesting the network after training:")
for inputs, actual in data:
    x1, x2 = inputs

    y1 = x1 * w11 + x2 * w21

    f_y1 = sigmoid(y1)

    y2 = x1 * w12 + x2 * w22
    f_y2 = sigmoid(y2)

    y_out = f_y1 * w_out1 + f_y2 * w_out2
    f_y_out = sigmoid(y_out)

    print("Input: ", inputs, "Predicted Output: ", round(f_y_out), "Actual Output: ", actual)
