import numpy as np


############# Explanation ##############

# Logistic Regression is a supervised learning model that is primarily used for binary classification problems.

# This is not a regression, is a classification model. The logistic regression model is used to predict the probability of a binary target variable.
# we use standard logistic function to model the probability of the target variable, also called the sigmoid function.

# If 𝜎(𝑧)>0.5  → Predicted class is 1.
# If 𝜎(𝑧)≤0.5 → Predicted class is 0.

# We maximize the likelihood of our training according to the model. 
# It's more conenient to mazimite log-likelihood, so we minimize the negative log-likelihood. 
# The negative log-likelihood is also called the cross-entropy loss.



class LogisticRegression:
    """
    Implementation of a Logistic Regression model
    Attributes:
    -----------
    weights : numpy.ndarray
        The weights of the logistic model.
    bias : float
        The bias term of the logistic model.
    Methods:
    --------
    __init__():
        Initializes the weights and bias to None.

    fit(X, y, epochs=1000, learning_rate=0.01):
        Trains the logistic regression model using the given training data.

    predict(X):
        Predicts the target values for the given input features.

    score(X, y):
        Returns the accuracy of the prediction.
    """

    def __init__(self):
        """Initiate weights and bias to None."""
        self.weights = None
        self.bias    = None

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """
        Train the logistic regression model using the gradient descent.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).
        y: numpy.ndarray
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

        # run gradient descent process

        for epoch in range(epochs):
            # Compute model prediction using sigmoid function
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # compute the gradient of the loss function
            error = y_predicted - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # update parameters
            self.weights = self.weights - learning_rate * dw
            self.bias    = self.bias    - learning_rate * db

            # Compute the loss (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9))

            # print the training loss at each epoch
            if verbose and epoch % 300 == 0:
                print(f'Epoch {epoch}, loss {loss}')

    def predict_prob(self, X):
        """
        Predict the probability of the target value for the given input using the trained model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).

        Returns:
        --------
        numpy.ndarray
            The predicted probability values of shape (n_samples,).
        """

        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict the target value for the given input using the trained model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).
        threshold : float, optional
            The threshold value to classify the target value (default is 0.5).

        Returns:
        --------
        numpy.ndarray
            The predicted target values of shape (n_samples,).
        """

        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        """
        Compute the accuracy of the model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).
        y : numpy.ndarray
            The target values of shape (n_samples,).

        Returns:
        --------
        float
            The accuracy of the model.
        """

        y_predicted = self.predict(X)
        accuracy    = np.mean(y_predicted == y)

        return accuracy

if __name__ == '__main__':
    # sample data for classification problem
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # train the model
    model = LogisticRegression()
    model.fit(X, y, epochs=1000, learning_rate=0.01)

    # make predictions
    y_proba = model.predict_prob(X)
    y_pred = model.predict(X)
    accuracy = model.score(X, y)

    print("\n-----------------------------")
    print(f"Input data:\n{X}")
    print(f"Actual labels: {y}")
    print(f"Predicted probabilities: {y_proba}")
    print(f"Predicted labels: {y_pred}")
    print(f"Model Accuracy: {accuracy:.2f}")
    print("-----------------------------")
    print("Logistic Regression model parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print("-----------------------------")