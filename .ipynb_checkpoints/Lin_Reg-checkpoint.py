import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
data = pd.read_csv("train.csv")
X = data['Size'].values
y = data['Price'].values

# Define the learning rate and number of epochs
learning_rate = 0.01
num_epochs = 100

# Initialize the model parameters (slope and intercept)
slope = 0
intercept = 0

# Lists to store the costs and accuracies after each epoch
costs = []
accuracies = []

# Implement linear regression with gradient descent
for epoch in range(num_epochs):
    y_pred = slope * X + intercept
    error = y_pred - y
    
    # Calculate the cost (mean squared error)
    cost = np.mean(error ** 2)
    costs.append(cost)
    
    # Calculate the gradient for the slope and intercept
    gradient_slope = (2 / len(y)) * np.dot(error, X)
    gradient_intercept = (2 / len(y)) * np.sum(error)
    
    # Update the model parameters
    slope -= learning_rate * gradient_slope
    intercept -= learning_rate * gradient_intercept
    
    # Calculate the accuracy (not applicable for linear regression, so we'll just use cost for reference)
    accuracy = cost
    accuracies.append(accuracy)
    
    print(f"Epoch {epoch + 1}: Cost = {cost}, Accuracy = {accuracy}")

# Plot the cost over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Over Epochs')
plt.show()

# Load the test data from "test.csv"
test_data = pd.read_csv("test.csv")
X_test = test_data['Size'].values

# Predict housing prices for the test data
y_pred_test = slope * X_test + intercept

# You can print or save y_pred_test as the predicted housing prices.

