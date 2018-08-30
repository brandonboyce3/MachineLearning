##########
###
### preProcessing.py
###
### Goal: To assist with Machine Learning projects by simplifying the pre-processing of test data
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
import numpy as np



##########
###
### preProcessGaussian()
###
### Purpose:
###    Pre-process the data to be a Gaussian of mean 0 and std 1
###
### Inputs:
###    data: Pandas DataFrame holding our data to be pre-processed
###
### Outputs:
###    data: Pandas DataFrame holding the pre-processed data
###
##########
def preProcessGaussian(data):
    ### Attempt to normalize the data such that it has a Gaussian distribution with mean 0 and std of 1
    for column in range(data.shape[1]):
        ### (x-x.mean)/x.std --> Should give you a Gaussian distribution of mean 0 and std 1
        data.iloc[:,column] = (data.iloc[:,column]-data.iloc[:,column].mean())/data.iloc[:,column].std()

        ### Quick check that things worked out (mean ~= 0 and std ~= 1)
        assert(data.iloc[:,column].mean()          < 1e-15)
        assert(np.abs(data.iloc[:,column].std()-1) < 1e-15)

    return data


