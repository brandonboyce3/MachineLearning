##########
###
### EM_GMM.py
###
### Goal: Run Expectation-Maximization of Gaussian Mixture Models (EM_GMM) on input data
###
### Terminology/algorithm obtained from: https://www.youtube.com/watch?v=qMTuMa86NzU (video from Prof. Alexander Ihler of UC-Irvine)
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
from matplotlib import pyplot as plt



##########
###
### EM_GMM()
###
### Purpose:
###    Run Expectation-Maximization of Gaussian Mixture Models (EM_GMM) on input data
###
### Inputs:
###    data       : Pandas DataFrame of new data on which to run EM_GMM
###    truth      : Pandas DataFrame of the classification for each row of the data
###    c          : Number of Gaussians in which to attempt to classify the data
###    do_plots   : Boolean on whether or not to plot things out
###    max_iters  : Maximum number of iterations that we should run before stopping
###
### Outputs:
###    new_data   : Pandas DataFrame of new data, that has reduced dimensionality
###
### Algorithm Overview:
###     Let X     = The matrix of data that we wish to run our EM_GMM algorithm on
###     Let x_i   = Data point "i"
###
###     Let c     = Total number of Gaussians
###     Let m     = Total number of data points
###     Let mu    = Mean              = (1/m)*sum_over_i( x_i )
###     Let Sigma = Covariance Matrix = (1/m)*sum_over_i( (x_i - mu)^T * (x_i - mu) )
###
###     Let r_ic    = Responsibility a given Gaussian "c" has over point "i" = (too complex for here, see link above)
###     Let m_c     = Total responsibility allocated to Gaussian "c"         = sum_over_i( r_ic )
###     Let pi_c    = "Weight" or fraction of "m" assigned to Gaussian "c"   = m_c/m
###     Let mu_c    = Mean of Gaussian "c"              = (1/m_c)*sum_over_i( r_ic*x_i )
###     Let Sigma_c = Covariance matrix of Gaussian "c" = (1/m_c)*sum_over_i( r_ic*( (x_i - mu_c)^T * (x_i - mu_c) ))
###
###     Initial values:
###         r_ic    = 0's
###         pi_c    = 1/c
###         mu_c    = Random data points in the input dataset
###         Sigma_c = Identity matrix
###
###     Let the "E-step" be:
###         Obtain new values for r_ic using the complicated equation (see link above)
###
###     Let the "M-step" be:
###         Obtain new values for m_c, pi_c, mu_c, and Sigma_c using the newly obtained r_ic values
###
###     Let us end our iterations of E-M when we hit a total number of iterations or our log_likelihood converges
###
##########
def EM_GMM(X, truth, c=-1, do_plots=True, max_iters=100, title='Plotting EM_GMM Results on input data'):
    ### Ensure that the user gave us a guess at number of Gaussians
    if(c <= 0):
        print 'Error: You did not specify a "c" value, i.e. number of Gaussians to use.'
        return

    ### Initialize variables
    m, d, mu, Sigma, pi, r = getInitialValues(X, c)

    ### If we want to draw the plots, start by creating a plot of the raw data points
    drawData_2D(do_plots, truth, title, 'initial', X, mu, r)

    ### Iterate until max_iters
    ### NOTE: In the future would want to add a condition about the log likelihood
    ###       For example, one could stop when the log likelihood stops incrementing by a percentage threshold
    for iter_count in range(max_iters):
        ### Run the E-step
        r = E_Step(X, c, m, d, mu, Sigma, pi, r)

        ### Run the M-step
        mu, Sigma, pi = M_Step(X, c, m, d, mu, Sigma, pi, r)

        ### Draw the progression of the means, for visually understanding the algorithm
        drawData_2D(do_plots, truth, title, 'during', X, mu, r)

    ### Draw the final location of the means
    drawData_2D(do_plots, truth, title, 'final', X, mu, r)

    ### Draw out our predictions
    drawData_2D(do_plots, truth, title, 'predict', X, mu, r)

    return X, mu, Sigma, pi



##########
###
### getInitialValues()
###
### Purpose:
###    Gets initial values for the EM_GMM algorithm
###
### Inputs:
###    X    : Pandas DataFrame of new data on which to run EM_GMM
###    truth: Pandas DataFrame of the classification for each row of the data
###    mu   : Pandas DataFrame, initial Means of each Gaussian cluster
###    r    : Pandas DataFrame, initial responsibility matrix
###
### Outputs:
###    Pretty Plots
###
##########
def drawData_2D(do_plots, truth, title, type, X, mu, r):
    ### If we didn't want to draw plots, then ignore this function
    if not do_plots:
        return

    ### Pre-define our color scheme (currently only supports 4 classes, but can update to make far better)
    unique = {name: [value, False] for value, name in enumerate(list(set(truth.iloc[:])))}
    marker = ['+', 'd', 's', 'h']
    colors = ['g', 'b', 'm', 'y']

    ### If this is the initial drawing step, then draw out all of the raw data
    if type == 'initial':
        for count, item in enumerate(X.values):
            if not unique[truth[count]][1]:
                plt.scatter(item[0], item[1], c='r', marker=marker[ unique[truth[count]][0] ], label=truth[count])
                unique[truth[count]][1] = True
            else:
                plt.scatter(item[0], item[1], c='r', marker=marker[ unique[truth[count]][0] ])
        for count, mean in enumerate(mu.values):
            plt.scatter(mean[0], mean[1], c=colors[count], marker='X')

    ### Plot out the progression of our means "mu_c"
    elif type == 'during':
        for count, mean in enumerate(mu.values):
            plt.scatter(mean[0], mean[1], c=colors[count], marker='.')

    ### Plot out the final location of our means "mu_c", and show the plot
    elif type == 'final':
        for count, mean in enumerate(mu.values):
            plt.scatter(mean[0], mean[1], c=colors[count], marker='*', s=100)
        plt.title(title)
        plt.legend()
        plt.show()

    ### Show our predictions for each data point
    elif type == 'predict':

        for count, item in enumerate(X.values):
            if not unique[truth[count]][1]:
                plt.scatter(item[0], item[1], c='r', marker=marker[ unique[truth[count]][0] ], label=truth[count])
                unique[truth[count]][1] = True
            else:
                plt.scatter(item[0], item[1], c='r', marker=marker[ unique[truth[count]][0] ])

        for ii in range(X.shape[0]):
            count = r.iloc[ii,:].idxmax()
            if not unique[truth[ii]][1]:
                plt.scatter(X.iloc[ii,0], X.iloc[ii,1], c=colors[count], marker=marker[ unique[truth[ii]][0] ], label=truth[ii])
                unique[truth[ii]][1] = True
            else:
                plt.scatter(X.iloc[ii,0], X.iloc[ii,1], c=colors[count], marker=marker[ unique[truth[ii]][0] ])
        plt.title(title)
        plt.legend()
        plt.show()

    else:
        print 'Error: Somehow got an unknown plotting type.'
        return



##########
###
### getInitialValues()
###
### Purpose:
###    Gets initial values for the EM_GMM algorithm
###
### Inputs:
###    X: Pandas DataFrame of new data on which to run EM_GMM
###    c: Number of Gaussian clusters to use
###
### Outputs:
###    m    : Number of data points
###    d    : Number of features/classes/dimensions
###    mu   : Pandas DataFrame, initial Means of each Gaussian cluster
###    Sigma: 3D numpy array  , initial Covariance Matrices of each Gaussian
###    pi   : Pandas DataFrame, initial weight matrix of each Gaussian
###    r    : Pandas DataFrame, initial responsibility matrix
###
##########
def getInitialValues(X, c):
    ### Get number of data points and number of dimensions
    m, d = X.shape

    ### Get initial values for mu by choosing random points
    mu = X.iloc[np.random.choice(m, c, False), :]

    ### Get initial values for Sigma by using the Identity matrix and knowing that Sigma should be (d,d) in shape
    Sigma = [np.eye(d)]*c

    ### Get initial values for pi knowing that every Gaussian should start off with equal weights
    pi = pd.DataFrame( [1./c]*c )

    ### Get initial values for r knowing that initially no Gaussian has any responsibility over any point
    r = pd.DataFrame( np.zeros( (m, c) ) )

    return m, d, mu, Sigma, pi, r



##########
###
### updateR()
###
### Purpose:
###    Update the "r" matrix using the value of the Gaussian distribution formula (N formula in link above)
###
### Inputs:
###    X    : Column of data associated with a single gaussian
###    c    : The index of the Gaussian that we are currently concerned with
###    m    : Number of data points
###    d    : Number of features/classes/dimensions
###    mu   : Mean of the Gaussian that we are dealing with
###    Sigma: Covariance Matrices of the Gaussian that we are dealing with
###    pi   : Weight value of the Gaussian that we are dealing with
###    r    : Pandas DataFrame, responsibility matrix that will be updated
###
### Outputs:
###    r    : Pandas DataFrame, updated responsibility matrix
###
##########
def updateR(X, c, m, d, mu, Sigma, pi, r):
    ### The Gaussian Distribution method is written out as such
    ### N(x|mu, Sigma) = ((2*pi)**(-d/2))*(Det(Sigma)**(-1/2))*(exp^( (-1/2)*(x-mu)^T*inv(Sigma)*(x-mu) ))
    ### Here, we split the formula into two parts, N_a and N_b, for ease of reading

    ### Calculate N_a, the part of the formula that is not within the exp()
    N_a = ((2.*np.pi)**(-d/2.))*(np.linalg.det(Sigma)**(-0.5))

    ### Calculate N_b, the part of the formula that is within the exp()
    ### NOTE: This should result in a single value, as x = (d,1), mu = (d,1), Sigma = (d,d)
    inv_Sigma = np.linalg.inv(Sigma)
    for ii in range(m):
        x_minus_mu  = X[ii,:] - mu
        dot_product = x_minus_mu.T.dot( inv_Sigma.dot(x_minus_mu) )
        N_b = np.exp( (-0.5)*dot_product )

        ### Update our "r" matrix in the appropriate location by finishing our calculation
        r.iloc[ii,c] = pi*N_a*N_b

    return r



##########
###
### E_Step()
###
### Purpose:
###    Run the "E-step" of the EM_GMM algorithm described in the description of EM_GMM()
###
### Inputs:
###    data        : Pandas DataFrame of new data on which to run EM_GMM
###
### Outputs:
###    new_data    : Pandas DataFrame of new data, that has reduced dimensionality
###
##########
def E_Step(X, c, m, d, mu, Sigma, pi, r):
    ### Iterate over each Gaussian, and update the "r" matrix using each Gaussian's mu, Sigma, and pi values
    for gaussian in range(c):
        r = updateR(X.values, gaussian, m, d, mu.iloc[gaussian,:].values, Sigma[gaussian], pi.iloc[gaussian].values, r)

    ### Ensure that we divide each value of r_ic by the total, as defined in the algorithm from the link above
    ### I.e. r_ic = (pi_c*N(x|mu_c,Sigma_c))/sum_over_c( pi_c*N(x|mu_c,Sigma_c) )
    r = (r.T/np.sum(r, axis=1)).T

    ### Return the updated responsibility matrix "r"
    return r



##########
###
### M_Step()
###
### Purpose:
###    Run the "M-step" of the EM_GMM algorithm described in the description of EM_GMM()
###
### Inputs:
###    data        : Pandas DataFrame of new data on which to run EM_GMM
###
### Outputs:
###    new_data    : Pandas DataFrame of new data, that has reduced dimensionality
###
##########
def M_Step(X, c, m, d, mu, Sigma, pi, r):
    ### Iterate over each Gaussian
    for gaussian in range(c):
        ### Update "m_c", the total responsibility associated to Gaussian "c"
        m_c = sum(r.iloc[:,gaussian])

        ### Update "pi_c", the "weight" of this Gaussian "c"
        pi.iloc[gaussian] = m_c/m

        ### Update "mu_c", the mean of Gaussian "c"
        mu.T.iloc[:,gaussian] = ((r.iloc[:,gaussian].T.dot(X.values))/m_c)

        ### Update "Sigma_c", the covariance matrix associated with Gaussian "c"
        new_Sigma = np.zeros( (d, d) )
        for ii in range(m):
            x_minus_mu = np.matrix( X.iloc[ii,:]-mu.iloc[gaussian,:] )
            new_Sigma += r.iloc[ii,gaussian]*( x_minus_mu.T.dot(x_minus_mu) )
        Sigma[gaussian] = new_Sigma/m_c

    ### Quick assert check that our "weights" approximately add up to 1
    assert(sum(pi.values) < 1+1e-10 and sum(pi.values) > 1-1e-10)

    ### Return the updated parameters
    return mu, Sigma, pi



##########
###
### proof_of_concept()
###
### Purpose:
###    Run EM-GMM Algorithm on 2D data that we know should be easily differentiable for a EM_GMM algorithm
###
### Inputs:
###    None
###
### Outputs:
###    Plots showing that we nailed it
###
##########
def proof_of_concept():
    ### Create random means, covariances, and sizes of each Gaussian
    mu    = [[-1, -4], [0, 3], [4, -2]]
    Sigma = [ [[0.3,  0.8], [ 0.2, 0.3]],
              [[0.1, -0.6], [-0.7, 0.3]],
              [[0.6,  0.1], [-0.4, 0.2]] ]
    sizes = [77, 44, 66]

    ### Create the new Gaussians using a specific "random seed" such that we get repeatable results for testing purposes
    X = 'initial'
    np.random.seed(7)
    for ii in range(len(sizes)):
        new_data = np.random.multivariate_normal(mu[ii], Sigma[ii], sizes[ii])
        if isinstance(X, str):
            X = new_data
        else:
            X = np.vstack( (X, new_data) )

    ### Ensure that we give each of the data points a "truth" label
    X = pd.DataFrame(X)
    X['2'] = pd.Series( ['a']*sizes[0] + ['b']*sizes[1] + ['c']*sizes[2] )

    ### Shuffle the data
    X = X.sample(frac=1).reset_index(drop=True)

    ### Test our algorithm
    EM_GMM(X.iloc[:,:-1], X.iloc[:,-1], 3, max_iters=25, title='Randomly Generated Gaussians to test our EM_GMM')



### Run our proof_of_concept
# proof_of_concept()


