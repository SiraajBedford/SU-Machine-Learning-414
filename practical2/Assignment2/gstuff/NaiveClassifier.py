import numpy as np
import math

class NaiveClassifier: # Accepts Row data

    def __init__(self): # constructor with implicit attributes (self.thing)
        self.__N = None # number of observations
        self.__d = None # number of dimensions in each observation
        self.__k = None # number of classes
        self.__classes = None # the list of class elements
        self.__class_Ns = None # the list of class counts
        self.__means = None 
        self.__variances = None
        self.__softmax_args = None # the softmax arguments a that are using in the softmax function

        # the variances are vectors of the diagomal elements on sigmas  

    def fit(self, X, y): # Accepts row data

        X = X.T # converts the row data to column data

        # X contains the observations as [x1, x2, ... xN]
        #       y contains the labels as [y1, y2, ... yN]

        self.__N = X.shape[1]  # layer1 is the columns (column data)
        self.__d = X.shape[0]  # layer0 is the rows
        self.__classes = list(set(y))
        self.__k = len(self.__classes) #number of classes = number of unique entries in y

        self.__calculate_means_and_variances(X, y)



    def predict(self, X): # Accepts row data
        # X contains test observations as [x1, x2, ... xQ]

        X = X.T # converts the row data to column data

        predict_N = X.shape[1]
        
        self.__softmax_args = np.zeros((self.__k, predict_N))
        labels = [0 for p in range(0, predict_N)]

        #x[p] is element p in X, and gets assigned a label[p]
        for p in range(0, predict_N):

            x = X[:,p].astype(np.float32)
            
            #initialise softmax parameters
            A = [0. for m in range(0, self.__k)]
            B = [0. for m in range(0, self.__k)]
        
            for m in range(0, self.__k):

                # A[m] = sum(0, l){ -0.5 * ln(2pi * var[l,m]) -1/(2 * var[l, m]) * (x[l, p] - mean[l, m])^2}
                for l in range(0, self.__d):
                    A[m] -= 0.5 * math.log(2 * math.pi * self.__variances[l, m])
                    A[m] -= (1/float(2 * (self.__variances[l, m]))) * ((x[l] - self.__means[l, m]) ** 2)

                # B[m] = ln(class_Ns[m]/N)
                B[m] = math.log(self.__class_Ns[m]/float(self.__N))

                # softmax_args[m] = A[m] + B[m]
                self.__softmax_args[m, p] = A[m] + B[m]


            # labels[p] = index_max{ a[p] } 
            # This is a simplified form of the argmax(softmax) that gives the same result
            labels[p] = self.__index_max(self.__softmax_args[:,p], self.__k)
            
        return np.array(labels)


        
    def __calculate_means_and_variances(self, X, y):

        # initialise all the variable shapes
        self.__class_Ns = [0. for n in range(0, self.__k)]        # contains [N1, N2, ... Nk]
        self.__means = np.zeros((self.__d, self.__k))            # contains [mean1, mean2, ... meank]
        self.__variances = np.zeros((self.__d, self.__k))        # contains [var1, var2, ... vark]

        
        # means[n] = scaled sum of elements xn in X, with associated label yn
        for m in range(0, self.__k):
            for n in range(0, self.__N):
                if y[n] == m:
                    x = X[:, n].astype(np.float32) # a valid data element x

                    self.__class_Ns[m] += 1
                    
                    for l in range(0, self.__d):
                        self.__means[l, m] += x[l]

        # normalises the mean vectors
        for m in range(0, self.__k):
            for l in range(0, self.__d):
                self.__means[l, m] /= self.__class_Ns[m]

        
        # variances[n] = the elements on the diagonal of sigmas[n]
        for m in range(0, self.__k):
            for n in range(0, self.__N):
                if y[n] == m:
                    cx = X[:, n].astype(np.float32)

                    for l in range(0, self.__d):
                        cx[l] -= self.__means[l,m]

                        self.__variances[l,m] += cx[l] ** 2


            self.__variances[:,m] /= float(self.__class_Ns[m])
            

        # This method of getting the covariance first and then the variances is unneccessary,
        # but its easier for me to implement

    def __index_max(self, arg, length):
        index_of_max = 0;
        maximum = arg[0]

        for m in range(0, length):
            if arg[m] > maximum:
                maximum = arg[m]
                index_of_max = m

        return index_of_max


    # Accessor methods to read private variables
    def get_N(self):
        return self.__N

    def get_d(self):
        return self.__d

    def get_k(self):
        return self.__k

    def get_classes(self):
        return self.__classes

    def get_class_Ns(self):
        return self.__class_Ns

    def get_means(self):
        return self.__means

    def get_variances(self):
        return self.__variances

    def get_softmax_args(self):
        return self.__softmax_args