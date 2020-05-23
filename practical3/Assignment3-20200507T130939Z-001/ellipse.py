# Import themes for the notebook
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import different modules for using with the notebook
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal as sig

# Import other useful suff
import matplotlib.cm as cee_em
from IPython.display import HTML
from IPython.display import display
from IPython.display import Image

# Some useful data science libraries
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal as mvn

from numpy.random import randint
from skimage import io
from ipywidgets import interact
from matplotlib.patches import Ellipse

import scipy as sc

# Settings for the notebook
%matplotlib inline
%load_ext autoreload
%autoreload 2





#DIY GMM




def KforG(X, k):
    # Generating a random startpoint, two cluster center points - Choose two random samples
    c = np.array(["r","g","b","y","c"])
    
    N=X.shape[1]
    d=X.shape[0]
    
    start_X_ind = np.random.randint(low=0, high=N, size = (k,d))
    #Mean = X[:,start_X_ind]
    Mean = np.zeros((k,d))

    for i in range(k):
        for j in range(d):
            Mean[i,j] = X[j,start_X_ind[i,j]]
    
    #Class labels
    label_ass = np.zeros(N,dtype=int) 
    
    #Perform 5 iterations of K-means
    runs = 1
    counter = 0
    d = np.zeros((k,N))
    while (runs == 1):
    #for counter in range(5):
        #Calculate Distances to cluster points
        for i in range(k):
            for j in range(N):           #all rows (dimensions), throgh N observations columns, - Means through k rows 
                d[i,j] = np.sqrt(np.sum((X[:,j]-Mean[i,:])**2))                                   # and all columns
                        
        #Assign a label to each observation depending to which cluster center it is the closest to

        label_ass = np.argmin(d, axis = 0)
        
        Mean_old = np.copy(Mean)
        
        #Calculate new cluster points; mean of all points belonging to the same cluster
        
        for i in range(k):
            Mean[i] = np.mean(X.T[label_ass == i], axis = 0)
            
        #Plot result
        
        counter+=1
        if (np.sqrt(np.sum((Mean-Mean_old)**2)) < 1e-2):
            runs =0
        
    cov = []
    nj = []
    for j in range(k):
        cov.append(np.cov(X[:,j==label_ass]))
        nj.append((X[:,j==label_ass]).shape[1])
        
        #Square old center star new center  
        #print(Mean)
        #print(Mean_old)
        
            
    return(Mean,np.array(cov),np.array(nj))  
    

def myGMM(X,k):
    means, covs, njs = KforG(X,k)
    N=X.shape[1]
    d=X.shape[0]
    label_ass = np.zeros(N,dtype=int)
    probs = np.zeros((k,N))
    
    for i in range(k): #Prior prob * gaussian dists
        probs[i] = (njs[i]/N)*sc.stats.multivariate_normal.pdf(X.T, means[i], covs[i])
    
    label_ass = np.argmax(probs, axis=0) 
    for j in range(k):  #new means
        means[j] = np.mean(X.T[label_ass==j], axis=0)
        covs[j] = np.cov(X[:,j==label_ass])
    
    c = np.array(["r","g","b","c","y", "m"])
    fig = plt.figure()
    axis = fig.add_subplot(111)
    
    for j in range(N):
        col = c[label_ass[j]]
        axis.scatter(X[0,j],X[1,j],c=col, zorder=1, alpha=0.4)
        
        
    eigvals, eigvecs = np.linalg.eig(covs)
    eigvals = np.sqrt(eigvals)
    for i in range(covs.shape[0]):
        ell = Ellipse(
            xy=(means[i]), 
            width=eigvals[i,0]*4, 
            height=eigvals[i,1]*4, 
            angle=-np.rad2deg(np.arccos(eigvecs[i]))[0,0],
            facecolor='None', 
            edgecolor=c[i]
        )
        axis.add_patch(ell) 
    plt.show()    
    




myGMM(X,2)


display(Image(filename='./gmm.jpg'))