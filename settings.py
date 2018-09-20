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

minibatching = False # Whether to minibatch
minibatchSize = 32

reluLayers = False

useFiles = False # If true, data is read from two files, one for interventions and one for base population.
dataFile = 'tasks/iris_interventions_removed.txt'
interventionsDataFile = 'tasks/iris_interventions.txt'

numInterventions = 10

# Settings if no file is used
numInputs = 32
numHidden = 16
numOutputs = 32
numHiddenLayers = 1
numTotalTasks = 5 # size of initial population, if file given, also applies
autoassociative = False

metric = "mae" # Which metric function to use -- goodness cannot be used for non-binary
                                # input datasets
metricFunction = None

minimumGoodness = 0.95
initialMinimumGoodness = minimumGoodness
dropout = 0.0
minimumMAE = 0.02


numExperiments = 15 # How many runs of the experiment, *not* repeats (need to rename)

stepSize = 2000 # The number of epochs to train before checking & printing
printRate = 2000 # How many steps to train for before printing
maxEpochs = 70000
initialMaxEpochs = 200000 # How many epochs to train for initially

