import numpy as np
import math

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


class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.positive = np.array([]) # probability for positive reviews
        self.negative = np.array([]) # probability for negative reviews
        self.p_positve = 0    # probablity of positive reviews
        self.p_negative = 0   # probablity of negative reviews
        
    def fit(self, X, Y):
        self.positive = np.full(len(X[0]),0.0) # Fill positive and negative arrays with 0s and prepare to add
        self.negative = np.full(len(X[0]),0.0)
        num_positve = 0
        num_negative = 0

        '''
        print("Length of X[0]: ", len(X[0]))
        print("-----------Positive--------")
        print(self.positive)
        print("Length of positive: ", len(self.positive))
        '''

        # Couting words and number of good/bad reviews
        for index in range(len(Y)):
            if Y[index] == 1:
                arr = self.positive
                num_positve += 1
            else:
                arr = self.negative
                num_negative += 1

            for x_index in range(len(X[index])):
                arr[x_index] = arr[x_index] + X[index][x_index]

        # Calculate Probability

        # thing_pos = [(i, float(self.positive[i] + 1.0), float(self.negative[i]) + 1.0) for i in range(len(self.positive))]
        # thing_pos.sort(key=lambda e: e[1] / e[2], reverse=True)
        # print(thing_pos[0:10])

        # for i, pos, neg in thing_pos:
        #     print()

        for x in range(len(self.positive)):
            p_prob = (self.positive[x] + 1)/(np.sum(self.positive) + len(self.positive))
            self.positive[x] = p_prob
            n_prob = (self.negative[x] + 1)/(np.sum(self.negative) + len(self.negative))
            self.negative[x] = n_prob
        
        self.p_positve = num_positve / len(Y)
        self.p_negative = num_negative / len(Y)

    def predict(self, X):
        Y = []
        for x in X:
            p_pos = math.log(self.p_positve,2)
            p_neg = math.log(self.p_negative,2)
            if len(x) != len(self.positive):
                print("Lengths of two sets are not equal!")
            for i in range(len(x)):
                p_pos += (math.log(self.positive[i], 2) * x[i])
                p_neg += (math.log(self.negative[i], 2) * x[i])
            
            if p_pos > p_neg:
                Y.append(1)
            else:
                Y.append(0)
        return np.array(Y)
                

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
        predictions = []
        for prob in y_predicted:
            if prob > 0.5:
               predictions.append(1)
            else:
                predictions.append(0)
        return predictions


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
