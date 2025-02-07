import numpy as np

############# Explanation ##############

# The general equation for a linear line is f = wx + b where f is the output, x is the input, w is the weight, and b is the bias.
# We want to find the optimal values for w and b that minimize the error between the predicted output and the actual output.
# For 1-D the equation is f = wx + b, for 2-D the equation is f = w1x1 + w2x2 + b, and so on.

# The solution for this problem is to minimize the loss function called Mean Squared Error (MSE) between the predicted output and the actual output.
# (f(xi) - yi)² where f(xi) is the predicted output and yi is the actual output.

class LinearRegression:
    """
    A simple Linear Regression model using Gradient Descent.

    Attributes:
    -----------
    weights : numpy.ndarray
        The weights of the linear model.
    bias : float
        The bias term of the linear model.

    Methods:
    --------
    fit(X, y, epochs=1000, learning_rate=0.01, verbose=True):
        Trains the linear regression model using the given training data.
    predict(X):
        Predicts the target values for the given input features.
    score(X, y):
        Returns the coefficient of determination R^2 of the prediction.
    """

    def __init__(self):
        """Initiate weights and bias to None."""
        self.weights = None
        self.bias    = None

    def fit(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """
        Trains the linear regression model using the given training data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).
        y : numpy.ndarray
            The target values of shape (n_samples,).
        epochs : int, optional
            The number of iterations to run the gradient descent algorithm.
        learning_rate : float, optional
            The learning rate for gradient descent (default is 0.01).
        verbose : bool, optional
            Whether to print the training loss at each epoch (default is True).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias    = 0

        # initiate the training process
        for epoch in range(epochs):

            # compute predictions with current weights and bias, equivalent to y=Xw+b
            y_predicted = np.dot(X, self.weights) + self.bias

            # we use gradient descent to adjust the weights and bias.
            # we take the average by dividing by n_samples
            error = y_predicted - y
            dw = (1 / n_samples) * np.dot(X.T, error) # we compute the partial derivative of the Mean Squared Error (MSE) with respect to the weights
            db = (1 / n_samples) * np.sum(error)      # we compute the partial derivative of the Mean Squared Error (MSE) with respect to the bias

            # update parameters
            self.weights = self.weights - learning_rate * dw
            self.bias    = self.bias    - learning_rate * db

            # compute loss
            loss = np.mean(np.square(error))

            # print loss every 100 epochs
            if verbose and (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}, loss {loss:.4f}")

    def predict(self, X):
        """
        Predicts the target value for the given input using the trained model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).

        Returns:
        --------
        numpy.ndarray
            The predicted target values of shape (n_samples,).
        """

        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """
        Compute the R-squared score of the model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).
        y : numpy.ndarray
            The target values of shape (n_samples,).

        Returns:
        --------
        float
            The R-squared score of the model.
        """

        # R2 is the percent of the variation of Y explained by X

        y_predicted = self.predict(X)
        ss_total    = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_predicted) ** 2)
        r2          = 1 - (ss_residual / ss_total)

        return r2


# Example

if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3  # y = 1*x1 + 2*x2 + 3

    # Train the model
    model = LinearRegression()
    model.fit(X, y, epochs=1000, learning_rate=0.01)

    # Make predictions
    y_pred = model.predict(X)
    r2     = model.score(X, y)

    print("-----------------------------")

    print(f"Input data: {X}")
    print(f"Actual values: {y}")
    print(f"Predictions: {y_pred}")
    print(f"R2 score: {r2}")

    print("-----------------------------")

    print("Regression model parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")

    print("-----------------------------")
