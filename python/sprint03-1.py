from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour


# Evaluate the linear regression
def compute_cost(X, y, theta, isPrint=False):
    '''
    Compute cost for linear regression
    '''
    # Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):
        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]  # X[:, 0] is all 1s
        errors_x2 = (predictions - y) * X[:, 1]

        # batch gradient
        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta, isPrint=True)

    return theta, J_history


# # Load the dataset
# data = loadtxt('ex1data1.txt', delimiter=',')
#
# # Plot the data
# scatter(data[:, 0], data[:, 1], marker='o', c='b')
# title('Profits distribution')
# xlabel('Population of City in 10,000s')
# ylabel('Profit in $10,000s')
# # show()
#
# X = data[:, 0]
# y = data[:, 1]
# # number of training samples
# m = y.size
# Add a column of ones to X (interception data)
# it = ones(shape=(m, 2))
# it[:, 1] = X

import pandas as pd
df_base = pd.read_csv("../data/house-prices-advanced-regression-techniques/train.csv")
df = df_base.loc[:, ["GrLivArea", "YearBuilt", "SalePrice"]]
feature_names = ["GrLivArea", "YearBuilt"]
y_name = "SalePrice"
x = df_base["GrLivArea"].values
y = df_base[y_name].values

# Initialize theta parameters
theta = zeros(shape=(2, 1))  # [[0.][0.]]
iterations = 3000
alpha = 0.01

# compute and display initial cost
compute_cost(x, y, theta)
theta, J_history = gradient_descent(x, y, theta, alpha, iterations)

print('theta is', theta)
# Predict values for population sizes of 35,000 and 70,000
predict1 = array([1, 3.5]).dot(theta).flatten()
# print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
# predict2 = array([1, 7.0]).dot(theta).flatten()
# print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))
#
# # Plot the results
# result = it.dot(theta).flatten()
# plot(data[:, 0], result)
# show()