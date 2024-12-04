# This code is for the visualisation of the decision Boundary by our Logistic Regression model 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
Bcancer = datasets.load_breast_cancer()
Bcancer_df = pd.DataFrame(Bcancer.data)
features = Bcancer.feature_names.copy()
Bcancer_df.columns = features
Bcancer_df['class'] = Bcancer.target

X = np.array(Bcancer_df.iloc[:, :-1])
y = np.array(Bcancer_df['class'])

# Add the bias term (1s column) to X
X = np.insert(X, 0, 1, axis=1)  # Insert 1's column for bias term

# Standard scaler to normalize the features
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

# Apply PCA to reduce to 2D for decision boundary visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X[:, 1:])

# Add the bias term (1s column) to reduced X for visualization
X_reduced = np.insert(X_reduced, 0, 1, axis=1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression function
def LogisticRegressor(X, y, max_iter, learn_rate):
    weights = np.random.randn(X.shape[1])  # Initialize weights

    for i in range(max_iter):
        for j in range(X.shape[0]):
            # Sigmoid output for each sample
            output = sigmoid(np.dot(X[j], weights))
            error = output - y[j]
            weights -= learn_rate * error * X[j]  # Update weights
    
    return weights

# Run logistic regression
weights = LogisticRegressor(X_reduced, y, max_iter=1000, learn_rate=0.01)

# Plotting the decision boundary for 2D data

# Create a mesh grid for the 2D plot
x_min, x_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
y_min, y_max = X_reduced[:, 2].min() - 1, X_reduced[:, 2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict the values using the logistic regression model
Z = sigmoid(np.dot(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], weights))
Z = Z.reshape(xx.shape)

# Plot the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.5)
plt.scatter(X_reduced[:, 1], X_reduced[:, 2], c=y, edgecolors='k', cmap="coolwarm")
plt.title('Decision Boundary and Data Points')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
