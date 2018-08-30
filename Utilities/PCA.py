##########
###
### PCA.py
###
### Goal: Run Principal Component Analysis on an input dataset to reduce dimensionality to an input number of dims
###
### Terminology/algorithm obtained from: https://en.wikipedia.org/wiki/Principal_component_analysis
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
from preProcessing import preProcessGaussian



##########
###
### PCA()
###
### Purpose:
###    Run Principal Component Analysis on the input data
###
### Inputs:
###    data        : Pandas DataFrame of new data on which to run PCA
###    des_variance: Int/float - [0, 100] - of the desired variance the user wants explained, to reduce dimensionality
###    has_labels  : Boolean telling us if the final column of new_data is the truth labels of each item
###    normalize   : Boolean on whether or not we should normalize the data first to be a Gaussian of mean 0 and std 1
###    user_dims   : A way for the user to override the number of dimensions found using the des_variance
###
### Outputs:
###    new_data    : Pandas DataFrame of new data, that has reduced dimensionality
###
##########
def PCA(data, des_variance=95, has_labels=True, normalize=True, user_dims=-1):
    ### Ensure that our des_variance is appropriate
    if( float(des_variance) < 0.0 or float(des_variance) > 100.0 ):
        print 'Error: Input desired variance to be explained is not within the range of [0, 100].'
        return data

    ### If we have labels, remove them from our data
    X = data
    if(has_labels):
        X = data.iloc[:,:-1]

    ### Start by normalizing the data if the user asked to
    if(normalize):
        X = preProcessGaussian(X)

    ### Get the covariance matrix, C
    ### NOTE: Using C = (1/(n-1))*(X-X_mean)^T*(X-X_mean)
    covariance_matrix = ((X-X.mean()).T.dot((X-X.mean())))/(X.shape[0]-1)

    ### Use the covariance matrix to get the eigenvector matrix, W, and eigenvalues, V
    ### NOTE: Using numpy as a shortcut here instead of doing my own eigenvalue decomposition - will update later
    eigenvalues, eigenvector_matrix = np.linalg.eig(covariance_matrix)

    ### Using the eigenvalues, determine how many dimensions we need to use to get to our desired explained variance
    ### To determine how much variance is explained, we divide the eigenvalue by the sum of eigenvalues
    ### We are also sorting the values from high to low, such that the we choose the minimum amount of dimensions
    cumulative_variance_explaned = np.cumsum([(ii/sum(np.abs(eigenvalues))) for ii in sorted(np.abs(eigenvalues), reverse=True)])
    print 'Cumulative variance explained: ', cumulative_variance_explaned
    if( user_dims > 0 and user_dims < len(eigenvalues) ):
        num_dims = user_dims
    else:
        num_dims = [idx for idx, val in enumerate(cumulative_variance_explaned) if val > float(des_variance)/100]
        num_dims = num_dims[0]+1 ### 0-based, so add 1, and should be protected since des_variance must be between [0, 100]

    ### Create the dimensionality reduced eigenvector matrix, W', based on the num_dims that we wanted to use
    ### Using "eigen pairs" to make it easier to sort things and create the W' matrix
    eigen_pairs = [(eigenvalues[ii], eigenvector_matrix[:,ii]) for ii in range(len(eigenvalues))]
    eigen_pairs = sorted(eigen_pairs, key=lambda pair: pair[0], reverse=True)
    eigenvector_matrix_dimensions_reduced = np.array([pair[1] for pair in eigen_pairs[0:num_dims]]).T

    ### Now, we use the W' matrix to project our data points onto the new feature space, T' = X*W'
    new_data = X.dot(eigenvector_matrix_dimensions_reduced)

    return new_data


