##########
###
### LDA.py
###
### Goal: Run Principal Component Analysis on an input dataset to reduce dimensionality to an input number of dims
###
### Terminology/algorithm obtained from: http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_LDA09.pdf
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
import pandas as pd
from preProcessing import preProcessGaussian



##########
###
### LDA()
###
### Purpose:
###    Run Linear Discriminant Analysis on the input data
###    NOTE: Assumes that the final column of the input data is the truth classifications of that data
###
### Inputs:
###    data        : Pandas DataFrame of new data on which to run PCA
###    des_variance: Int/float - [0, 100] - of the desired variance the user wants explained, to reduce dimensionality
###    normalize   : Boolean on whether or not we should normalize the data first to be a Gaussian of mean 0 and std 1
###    user_dims   : A way for the user to override the number of dimensions found using the des_variance
###
### Outputs:
###    new_data    : Pandas DataFrame of new data, that has reduced dimensionality
###
##########
def LDA(data, des_variance=95, normalize=True, user_dims=-1):
    ### Start by determining which unique classes we have
    unique_classes = list(set(data.iloc[:,-1]))

    ### Start by normalizing the data if the user asked to
    if(normalize):
        data.iloc[:,:-1] = preProcessGaussian(data.iloc[:,:-1])

    ### Determine the means of each class, and create "class_means"
    class_means = np.zeros( (data.shape[1]-1, len(unique_classes)) )
    for new_class in range(len(unique_classes)):
        class_means[:,new_class] = np.mean( data[ data.iloc[:,-1]==unique_classes[new_class] ].iloc[:,:-1] ).values

    ### Update the class means to be a Pandas DataFrame matrix
    class_means = pd.DataFrame(class_means, columns=unique_classes)

    ### Get the "within-class" scatter matrix
    within_class_scatter = np.zeros( (class_means.shape[0], class_means.shape[0]) )
    for new_class in range(len(unique_classes)):
        ### Get the data associated with the current class
        current_class_data = data[data.iloc[:, -1] == unique_classes[new_class]].iloc[:, :-1].values

        ### Subtract the class means from each row
        current_class_data = np.array([new_row - class_means.iloc[:,new_class].values for new_row in current_class_data])

        ### Get the scatter matrix associated with the current class
        current_class_scatter = (current_class_data.T.dot(current_class_data))/(current_class_data.shape[0]-1)

        ### Update the "within-class" scatter matrix
        within_class_scatter += current_class_scatter

    ### Get the "between-class" scatter matrix
    between_class_scatter = np.zeros( (class_means.shape[0], class_means.shape[0]) )
    for new_class in range(len(unique_classes)):
        ### Get the data associated with the current class
        current_class_data = np.matrix( class_means.iloc[:, new_class].values - data.iloc[:,:-1].mean() )

        ### Get the scatter matrix associated with the current class
        current_class_scatter = (current_class_data.T.dot(current_class_data))

        ### Add the scatter matrix to the current, and ensure we multiply by the number of points in the class
        between_class_scatter += current_class_scatter*current_class_data.shape[1]

    ### Use the scatter matrices to get the eigenvector matrix, W, and eigenvalues, V
    ### NOTE: Using numpy as a shortcut here instead of doing my own eigenvalue decomposition - will update later
    eigenvalues, eigenvector_matrix = np.linalg.eig(np.linalg.inv(within_class_scatter).dot(between_class_scatter))

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
    new_data = data.iloc[:,:-1].values.dot(eigenvector_matrix_dimensions_reduced)

    return pd.DataFrame(new_data)


