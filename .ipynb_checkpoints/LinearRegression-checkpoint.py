import pandas as pd
import matplotlib.pyplot as plt

# Load the data from "train.csv"
data = pd.read_csv("train.csv")

# Define the gradient descent function
def gradient_descent(w, b, data, learning_rate):
    m = len(data)
    w_gradient = 0
    b_gradient = 0

    for i in range(m):
        x = data.iloc[i]["sqft"]  # Corrected the variable assignment
        y = data.iloc[i]["price"]  # Corrected the variable assignment
        error = w * x + b - y
        w_gradient += error * x
        b_gradient += error

    w -= (1 / m) * w_gradient * learning_rate
    b -= (1 / m) * b_gradient * learning_rate

    return w, b

# Initialize weights and hyperparameters
w = 0
b = 0
learning_rate = 0.00001
epochs = 1000

# Perform gradient descent
for i in range(epochs):
    if i % 50 == 0:
        print("Epoch:", i)
    w, b = gradient_descent(w, b, data, learning_rate)

# Plot the data and regression line
plt.scatter(data["sqft"], data["price"])
xplot = list(data["sqft"])
yplot = [w * x + b for x in xplot]
plt.plot(xplot, yplot)
plt.show()

