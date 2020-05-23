import numpy as np
import math

class LogisticRegression:

    # constructor with implicit attributes
    def __init__(self, hyper, tolerance):
        self.__hyper = hyper
        self.__tolerance = tolerance

        self.__N = None # number of observations
        self.__d = None # number of dimensions in each observation
        self.__k = None # number of classes
        self.__w = None # the optimized decision boundary parameter

    def fit(self, X, y):

        X = X.T # converts the row data to column data

        # X contains the observations as [x1, x2, ... xN]
        #       y contains the labels as [y1, y2, ... yN]

        self.__N = X.shape[1]  # layer1 is the columns (column data)
        self.__d = X.shape[0]  # layer0 is the rows
        self.__classes = list(set(y))
        self.__k = len(self.__classes) #number of classes = number of unique entries in y

        # modify data with an additional row of ones
        ones_row = [1 for n in range(0, self.__N)]
        X = np.vstack([ones_row, X])

        #initialise w
        w = np.ones((self.__d + 1, 1))

        # measure = mag(w[n + 1] - w[n])/mag(w[n])
        measure = 1000;  # just a large starting value

        # Update Rule: w[n + 1] = w[n] - [H]^(-1) * G(w[n])
        
        # until measure < tolerance
        while measure >= self.__tolerance:
            w_prev = w

            w = w - np.linalg.inv(self.__hessian(X, w)).dot(self.__gradient(X, y, w))
            measure = np.linalg.norm(w - w_prev)/np.linalg.norm(w_prev)

        self.__w = w


    def predict(self, X):
        
        X = X.T # converts the row data to column data

        # X contains the observations as [x1, x2, ... xQ]

        predict_N = X.shape[1]
        labels = [0 for p in range(0, predict_N)]

        # modify data with an additional row of ones
        ones_row = [1 for n in range(0, predict_N)]
        X = np.vstack([ones_row, X])

        for p in range(0, predict_N):

            x = X[:, p][:, np.newaxis].astype(np.float32)
            sig = self.__sigmoid(self.__w, x)

            if sig >= 0.5:
                labels[p] = 1  # no else case needed since the labels are initialised to 0

        return np.array(labels)


    def __hessian(self, X, w):

        #initialise hessian
        hess = 0

        # hess = sum(0, N){ sigmoid * (1 - sigmoid) * outer_x) + (1/hyper) * I }
        for n in range(0, self.__N):

            x = X[:, n][:, np.newaxis].astype(np.float32)
            sig = self.__sigmoid(w, x)
            outer_x = x.dot(x.T)

            hess += sig * (1 - sig) * outer_x + (1/self.__hyper) * np.identity(self.__d + 1)

        return hess

    def __gradient(self, X, y, w):

        #initialise gradient
        grad = 0

        #grad = sum(0, N){ (sigmoid - y[n]) * x }   +   (1/hyper) * w 
        for n in range(0, self.__N):

            x = X[:, n][:, np.newaxis].astype(np.float32)
            sig = self.__sigmoid(w, x)

            grad += (sig - y[n]) * x

        # include the regularisation term  afterwards 
        grad += (1/self.__hyper) * w

        return grad

    def __sigmoid(self, w, x):

        # sigmoid = sigmoid{ w.T.dot(x[n]) }
        arg = w.T.dot(x)
        sigmoid = 1/float(1 + np.exp(-arg))

        return sigmoid
