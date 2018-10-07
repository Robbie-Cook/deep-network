"""
Settings.py -- Project settings
"""

import tensorflow as tf

"""
Settings
"""

method = 'pseudoSweep' # sweep, pseudoSweep, catastrophicForgetting
bufferSize = 4 # Buffer size for sweep and pseudosweep
bufferRefreshRate = 1 # How often to refresh the buffer
auxNetwork = False # Whether to use the autoencoder to generate psuedoinputs

modelType = 'adam' # Which training loss and optimizer to use -- options are 'adam','classification','normal'
numClasses = 10 # if classification

networkInputType = 'files' # Whether to use random generated input or structured input from a file
                                        # Options are 'randomGenerated' or 'files'

metric = 'mae' # Which metric function to use -- goodness cannot be used for non-binary
                    # Options are: 'mae', 'goodness', 'accuracy'

minimumGoodness = 0.95
initialMinimumGoodness = minimumGoodness
dropout = 0.0
minimumMAE = 0.05
minimumAccuracy = 0.95



taskFile = 'tasks/winequality-red.txt'#'tasks/iris_initial.txt'
interventionsFile = 'randomTask' # if set to 'randomTask', a random binary interventions are generated

useRanges = True # Whether to use ranges to generate pseudoitems
ranges = True # dont touch -- done automatically in main.py

reluLayers = False
dropout = 0.0

numInputs = 11 # How many inputs there are to the network
numHidden = 32
numOutputs = 1 # How many outputs the network has. For classification, this is the number of classes there are
numHiddenLayers = 1

basePopulationSize = 20 # size of initial population, if file given, also applies
numInterventions = 10
autoassociative = False

numExperiments = 50 # How many runs of the experiment, *not* repeats (need to rename)

stepSize = 100 # The number of epochs to train before checking & printing
printRate = 100 # How many steps to train for before printing
maxEpochs = 20000
initialMaxEpochs = 20000 # How many epochs to train for initially

