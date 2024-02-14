import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # Sigmoid function

def calculate_fpr(y_true, y_pred, group):
    """
    Calculate the False Positive Rate (FPR) for each group.

    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels.
    :param group: Array indicating group membership (0 or 1).
    :return: FPR for group 0 and group 1.
    """
    # Confusion matrix for each group
    cm_group0 = confusion_matrix(y_true[group == 0], y_pred[group == 0])
    cm_group1 = confusion_matrix(y_true[group == 1], y_pred[group == 1])
    # FPR calculation for each group
    fpr_group0 = cm_group0[0, 1] / (cm_group0[0, 1] + cm_group0[0, 0])
    fpr_group1 = cm_group1[0, 1] / (cm_group1[0, 1] + cm_group1[0, 0])

    return fpr_group0, fpr_group1



class FairLogisticRegression():
    def __init__(self, alpha=0.01, fairness_penalty=1.0, max_iter=100, tol=1e-3):
        """
        Fair Logistic Regression with a fairness penalty term.

        :param alpha: Learning rate.
        :param fairness_penalty: Weight for the fairness penalty term.
        :param max_iter: Maximum number of iterations for gradient descent.
        :param tol: Tolerance for stopping criteria.
        """
        self.alpha = alpha
        self.fairness_penalty = fairness_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def fit(self, X, y, group):
        """
        Fit the logistic regression model with fairness constraints.

        :param X: Feature matrix.
        :param y: Target vector.
        :param group: Array indicating group membership (0 or 1).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            scores = np.dot(X, self.weights)
            predictions = expit(scores)  # Sigmoid function to get probabilities

            # Gradient of the logistic loss
            gradient = np.dot(X.T, predictions - y) / n_samples

            # Calculate FPR for each group and its gradient
            fpr_group0, fpr_group1 = calculate_fpr(y, (predictions > 0.5).astype(int), group)
            fpr_diff = fpr_group0 - fpr_group1
            fpr_gradient = np.dot(X.T, group * (predictions - y)) / n_samples

            # Update weights with fairness penalty
            self.weights -= self.alpha * (gradient + self.fairness_penalty * fpr_diff * fpr_gradient)

            # Check convergence
            if np.linalg.norm(self.alpha * gradient) < self.tol:
                break

    def predict(self, X):
        """
        Predict using the logistic regression model.

        :param X: Feature matrix.
        :return: Predicted labels.
        """
        scores = np.dot(X, self.weights)
        return (expit(scores) > 0.5).astype(int)

