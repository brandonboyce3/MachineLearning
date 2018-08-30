##########
###
### cryo.py
###
### Goal: Import the cryo dataset in a manner that we have standardized for ourselves
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
###    Parse the "cryo.csv" dataset and put it into variables that we understand
###
### Inputs:
###    None
###
### Outputs:
###    cryo_data: Pandas DataFrame holding parsed version of "cryo.csv"
###
##########
def getData():
    ### Get the dataset via the .csv file
    ### NOTE: This dataset is nice and complete, so we don't have to worry about N/A or anything special here
    cryo_data = pd.read_csv(filepath_or_buffer=os.path.dirname(__file__) + '/../Data/Cryotherapy.csv', sep=',')

    ### Be sure to return the cryo_data
    return cryo_data



##########
###
### analysisPCA()
###
### Purpose:
###    Analyze the results of passing the cryo dataset to our and the sklearn PCA algorithms
###
### Inputs:
###    cryo_data: Pandas DataFrame holding parsed version of "cryo.csv"
###
### Outputs:
###    Plots comparing our PCA algorithm and the sklearn PCA algorithm
###
##########
def analysisPCA(cryo_data, normalize=True):
    ### Get results on my own PCA on this dataset
    new_data = PCA(cryo_data, normalize=normalize)
    plotResults_2D(new_data, cryo_data.iloc[:,-1], 'Custom PCA Results on cryo Dataset - Normalized = '+str(normalize))

    ### Get results to compare to using the sklearn version of PCA on this dataset
    pca = sklearn_PCA(n_components=2)
    if normalize:
        sklearn_data = sklearn_SS().fit_transform(cryo_data.iloc[:,:-1])
        sklearn_new_data = pca.fit_transform(sklearn_data)
    else:
        sklearn_new_data = pca.fit_transform(cryo_data.iloc[:,:-1])
    plotResults_2D(pd.DataFrame(sklearn_new_data), cryo_data.iloc[:,-1], 'Sklearn PCA Results on cryo Dataset - Normalized = '+str(normalize))



##########
###
### analysisLDA()
###
### Purpose:
###    Analyze the results of passing the cryo dataset to our and the sklearn LDA algorithms
###
### Inputs:
###    cryo_data: Pandas DataFrame holding parsed version of "cryo.csv"
###
### Outputs:
###    Plots comparing our LDA algorithm and the sklearn LDA algorithm
###
##########
def analysisLDA(cryo_data, normalize=True):
    ### Get results on my own PCA on this dataset
    new_data = LDA(cryo_data, user_dims=2, normalize=normalize)
    plotResults_2D(new_data, cryo_data.iloc[:,-1], 'Custom LDA Results on cryo Dataset - Normalized = '+str(normalize))

    ### Get results to compare to using the sklearn version of LDA on this dataset
    lda = sklearn_LDA(n_components=2)
    if normalize:
        sklearn_data     = sklearn_SS().fit_transform(cryo_data.iloc[:,:-1])
        sklearn_new_data = lda.fit_transform(sklearn_data, cryo_data.iloc[:,-1])
    else:
        sklearn_new_data = lda.fit_transform(cryo_data.iloc[:,:-1], cryo_data.iloc[:,-1])
    plotResults_2D(pd.DataFrame(sklearn_new_data), cryo_data.iloc[:,-1], 'Sklearn LDA Results on cryo Dataset - Normalized = '+str(normalize))



##########
###
### analysisEM_GMM()
###
### Purpose:
###    Analyze the results of passing the cryo dataset to our EM_GMM algorithm
###
### Inputs:
###    cryo_data: Pandas DataFrame holding parsed version of "cryo.csv"
###
### Outputs:
###    Plots comparing our EM_GMM algorithm
###
##########
def analysisEM_GMM(cryo_data, use_PCA=True, normalize=True, title="EM_GMM Results"):
    ### Define a seed that doesn't push two Gaussians right next to each other
    np.random.seed(1)

    ### Reduce dimensionality to 2D
    new_data = []
    if use_PCA:
        new_data = PCA(cryo_data, normalize=normalize)
    else: ### use_LDA
        new_data = LDA(cryo_data, user_dims=2, normalize=normalize)

    ### Run the EM_GMM algorithm to attempt to classify our data points
    EM_GMM(new_data, cryo_data.iloc[:,-1], 2, max_iters=10, title=title)



##########
###
### __main__()
###
### Purpose:
###    This will be called when this script is run, and will call all of our functions appropriately for debugging
###
##########
if __name__ == '__main__':
    ### Get data
    cryo_data = getData()
    analysisEM_GMM(cryo_data, False, True, "Cryo EM_GMM after LDA (Normalized)")
