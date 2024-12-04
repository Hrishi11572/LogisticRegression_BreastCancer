# Logistic Regression for Breast Cancer Classification

This project implements a **Binary Logistic Regression** model using a custom implementation of the algorithm, applied to the **Breast Cancer dataset** from sklearn. The model predicts whether a given tumor is malignant or benign based on various features extracted from digitized images of a breast mass.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Getting Started](#getting-started)
5. [Implementation](#implementation)
6. [Results](#results)
7. [Visualization](#visualization)
8. [License](#license)

---

## Overview
This project builds a logistic regression classifier from scratch to predict the class of breast cancer tumors (benign or malignant) based on 30 different features. The goal is to minimize the binary cross-entropy (log loss) using gradient descent.

The model uses **sigmoid** activation for binary classification and performs training through **stochastic gradient descent** to update the weights iteratively. The loss curve is plotted for visualization of model convergence.

## Features
- **Breast Cancer Dataset**: The model uses the `load_breast_cancer()` dataset from sklearn, which contains 569 samples and 30 features, plus a target variable indicating whether the tumor is malignant or benign.
- **Data Standardization**: The feature values are standardized using `StandardScaler` for better convergence during training.
- **Sigmoid Activation Function**: The model employs the sigmoid function to map input features to a probability between 0 and 1.
- **Gradient Descent**: The weights are updated using the gradient of the binary cross-entropy loss function.
- **Loss Curve Visualization**: The loss curve is plotted to show how the model's performance improves during training.

## Requirements
To run this code, ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```
## Getting Started

1. Clone the repository to your local machine:
   
   ``` bash
   git clone https://github.com/yourusername/logistic-regression-breast-cancer.git
   ```
2. Navigate into the project directory:
   
   ``` bash
   cd logistic-regression-breast-cancer
   ```
3. Run the Python script:
   
   ``` bash
   python logistic_regression.py
   ```
   
## Implementation

The script performs the following steps:

<b>Data Loading:</b> The `load_breast_cancer()` function from sklearn loads the Breast Cancer dataset. <br>

<b>Data Preprocessing:</b> The data is standardized using StandardScaler. <br>

<b>Logistic Regression Model:</b>
1. A custom function `LogisticRegressor` is defined to perform training using stochastic gradient descent (SGD).
2. The sigmoid function is applied to predict probabilities.
3. The weights are updated using the gradient of the loss function (binary cross-entropy).

<b>Loss Calculation:</b> The binary cross-entropy loss is calculated and recorded after each iteration.

<b>Visualization:</b> A loss curve is plotted to visualize the model’s progress over iterations.


<I>The code : `binary_logistic_with_PCA.py` uses PCA to plot the decision boundary using 2 features.</I>
   
   
