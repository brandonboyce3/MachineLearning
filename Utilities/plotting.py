##########
###
### plotting.py
###
### Goal: Plot out the data that we have manipulated using Machine Learning techniques
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
from matplotlib import pyplot as plt



##########
###
### plotResults()
###
### Purpose:
###    To plot out the results of the changes that we've applied to the dataset
###
### Inputs:
###    data : Pandas DataFrame of the data
###    truth: Pandas DataFrame of the classification for each row of the data
###    title: Title for the new plot that we are making
###
### Outputs:
###    Beautiful plots (currently only 2D is supported)
###
##########
def plotResults_2D(data, truth, title):
    ### Preset colors for each cluster, and get a list of our unique classes
    colors = ['b', 'r', 'm', 'g', 'c', 'y', 'k']
    unique = list(set(truth.iloc[:]))
    colors = {unique[ii]: colors[ii] for ii in range(len(unique))}

    ### Plot things out for us to see
    for ii in range(data.shape[0]):
        ### Get the new class and new color
        new_class = truth.iloc[ii]
        new_color = colors[new_class]

        ### Plot with a label if it is a new instance of a class, otherwise, just plot it
        if new_class in unique:
            unique.remove(new_class)
            plt.scatter(data.iloc[ii, 0], data.iloc[ii, 1], c=new_color, marker='o', label=new_class)
        else:
            plt.scatter(data.iloc[ii, 0], data.iloc[ii, 1], c=new_color, marker='o')

    ### Add our legend and show
    plt.title(title)
    plt.legend()
    plt.show()


