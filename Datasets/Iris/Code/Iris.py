##########
###
### Iris.py
###
### Goal: Import the Iris dataset in a manner that we have standardized for ourselves
###
### Updates:
###     08-29-2018: Initial creation
###
##########

###############
###
### Imports
###
###############
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as sklearn_SS
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklearn_LDA

### Custom imports
sys.path.append(os.path.dirname(__file__) + '/../../../Utilities/')
from plotting import plotResults_2D
from PCA import PCA
from LDA import LDA
from EM_GMM import EM_GMM



##########
###
### getData()
###
### Purpose:
###    Parse the "iris.csv" dataset and put it into variables that we understand
###
### Inputs:
###    None
###
### Outputs:
###    iris_data: Pandas DataFrame holding parsed version of "iris.csv"
###
##########
def getData():
    ### Get the dataset via the .csv file
    ### NOTE: This dataset is nice and complete, so we don't have to worry about N/A or anything special here
    iris_data = pd.read_csv(filepath_or_buffer=os.path.dirname(__file__) + '/../Data/iris.csv', sep=',')

    ### Be sure to return the iris_data
    return iris_data



##########
###
### analysisPCA()
###
### Purpose:
###    Analyze the results of passing the iris dataset to our and the sklearn PCA algorithms
###
### Inputs:
###    iris_data: Pandas DataFrame holding parsed version of "iris.csv"
###
### Outputs:
###    Plots comparing our PCA algorithm and the sklearn PCA algorithm
###
##########
def analysisPCA(iris_data, normalize=True):
    ### Get results on my own PCA on this dataset
    new_data = PCA(iris_data, normalize=normalize)
    plotResults_2D(new_data, iris_data.iloc[:,-1], 'Custom PCA Results on Iris Dataset - Normalized = '+str(normalize))

    ### Get results to compare to using the sklearn version of PCA on this dataset
    pca = sklearn_PCA(n_components=2)
    if normalize:
        sklearn_data = sklearn_SS().fit_transform(iris_data.iloc[:,:-1])
        sklearn_new_data = pca.fit_transform(sklearn_data)
    else:
        sklearn_new_data = pca.fit_transform(iris_data.iloc[:,:-1])
    plotResults_2D(pd.DataFrame(sklearn_new_data), iris_data.iloc[:,-1], 'Sklearn PCA Results on Iris Dataset - Normalized = '+str(normalize))



##########
###
### analysisLDA()
###
### Purpose:
###    Analyze the results of passing the iris dataset to our and the sklearn LDA algorithms
###
### Inputs:
###    iris_data: Pandas DataFrame holding parsed version of "iris.csv"
###
### Outputs:
###    Plots comparing our LDA algorithm and the sklearn LDA algorithm
###
##########
def analysisLDA(iris_data, normalize=True):
    ### Get results on my own PCA on this dataset
    new_data = LDA(iris_data, user_dims=2, normalize=normalize)
    plotResults_2D(new_data, iris_data.iloc[:,-1], 'Custom LDA Results on Iris Dataset - Normalized = '+str(normalize))

    ### Get results to compare to using the sklearn version of LDA on this dataset
    lda = sklearn_LDA(n_components=2)
    if normalize:
        sklearn_data     = sklearn_SS().fit_transform(iris_data.iloc[:,:-1])
        sklearn_new_data = lda.fit_transform(sklearn_data, iris_data.iloc[:,-1])
    else:
        sklearn_new_data = lda.fit_transform(iris_data.iloc[:,:-1], iris_data.iloc[:,-1])
    plotResults_2D(pd.DataFrame(sklearn_new_data), iris_data.iloc[:,-1], 'Sklearn LDA Results on Iris Dataset - Normalized = '+str(normalize))



##########
###
### analysisEM_GMM()
###
### Purpose:
###    Analyze the results of passing the iris dataset to our EM_GMM algorithm
###
### Inputs:
###    iris_data: Pandas DataFrame holding parsed version of "iris.csv"
###
### Outputs:
###    Plots comparing our EM_GMM algorithm
###
##########
def analysisEM_GMM(iris_data, use_PCA=True, normalize=True, title="EM_GMM Results"):
    ### Define a seed that doesn't push two Gaussians right next to each other
    np.random.seed(5)

    ### Reduce dimensionality to 2D
    new_data = []
    if use_PCA:
        new_data = PCA(iris_data, normalize=normalize)
    else: ### use_LDA
        new_data = LDA(iris_data, user_dims=2, normalize=normalize)

    ### Run the EM_GMM algorithm to attempt to classify our data points
    EM_GMM(new_data, iris_data.iloc[:,-1], 3, max_iters=50, title=title)



##########
###
### __main__()
###
### Purpose:
###    This will be called when this script is run, and will call all of our functions appropriately for debugging
###
##########
if __name__ == '__main__':
    ### Get the data
    iris_data = getData()

    ### Run PCA analysis
    analysisPCA(iris_data, True)
    analysisPCA(iris_data, False)

    ### Run LDA analysis
    analysisLDA(iris_data, True)
    analysisLDA(iris_data, False)

    ### Run EM_GMM analysis
    analysisEM_GMM(iris_data, True , True , "PCA - Normalized - EM_GMM")
    analysisEM_GMM(iris_data, False, True , "LDA - Normalized - EM_GMM")
    analysisEM_GMM(iris_data, True , False, "PCA - Not Normalized - EM_GMM")
    analysisEM_GMM(iris_data, False, False, "LDA - Not Normalized - EM_GMM")


