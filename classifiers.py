import numpy as np

# You need to build your own model here instead of using existing Python
# packages such as sklearn!

## But you may want to try these for comparison, that's fine.
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
              is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where
              N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
            is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the
            number of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
        

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")

# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self,learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1+ np.exp(-z))
    
    def fit(self, X, Y):

        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)

        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):
            # Linear combination of inputs and weights
            linear_model = np.dot(X, self.weights) + self.bias

            # using sigmoid function for the predictions 
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - Y))
            db = (1 / num_samples) * np.sum(y_predicted - Y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
       
       linear_model = np.dot(X, self.weights) + self.bias
       y_predicted = self.sigmoid(linear_model)

       for prob in y_predicted:
           if prob > 0.5: return 1 
           else: return 0


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
