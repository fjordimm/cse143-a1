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
        self.positive = np.array([]) # probability for positive reviews
        self.negative = np.array([]) # probability for negative reviews
        self.num_positve = 0    # number of positive reviews
        self.num_negative = 0   # number of negative reviews
        
    def fit(self, X, Y):
        # Add your code here!
        self.positive = np.full(len(X[0]),0) # Fill positive and negative arrays with 0s and prepare to add
        self.negative = np.full(len(X[0]),0)
        
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
                self.num_positve += 1
            else:
                arr = self.negative
                self.num_negative += 1

            for x_index in range(len(X[index])):
                arr[x_index] = arr[x_index] + X[index][x_index]

        #print(self.positive)
        #print(self.negative) 



    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")

# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
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


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
