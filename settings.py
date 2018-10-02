"""
Settings.py -- Project settings
"""

import tensorflow as tf

"""
Settings
"""

method = 'catastrophicForgetting' # sweep, pseudoSweep, catastrophicForgetting
bufferSize = 4 # Buffer size for sweep and pseudosweep
bufferRefreshRate = 1 # How often to refresh the buffer


modelType = 'normal' # Which training loss and optimizer to use -- options are 'adam','classification','normal'
numClasses = 10 # if classification

networkInputType = 'randomGenerated' # Whether to use random generated input or structured input from a file
                                        # Options are 'randomGenerated' or 'files'

metric = 'mae' # Which metric function to use -- goodness cannot be used for non-binary
                    # Options are: 'mae', 'goodness', 'accuracy'

minimumGoodness = 0.95
initialMinimumGoodness = minimumGoodness
minimumMAE = 0.02
minimumAccuracy = 0.95

taskFile = 'tasks/winequality-red.txt'#'tasks/iris_initial.txt'
interventionsFile = 'tasks/winequality-red.txt'#'tasks/iris_interventions.txt'
ranges = None

reluLayers = True
dropout = 0.0

numInputs = 32 # How many inputs there are to the network
numHidden = 16
numOutputs = 32 # How many outputs the network has. For classification, this is the number of classes there are


numHiddenLayers = 1
basePopulationSize = 5 # size of initial population, if file given, also applies
numInterventions = 10
autoassociative = False

numExperiments = 20 # How many runs of the experiment, *not* repeats (need to rename)

stepSize = 700 # The number of epochs to train before checking & printing
printRate = 700 # How many steps to train for before printing
maxEpochs = 70000
initialMaxEpochs = 200000 # How many epochs to train for initially

