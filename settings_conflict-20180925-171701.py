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
reluLayers = True



modelTypes =  ['classification', 'adam', 'normal']
modelType = 'classification' # Which training loss and optimizer to use -- options are 'adam','classification','normal'
assert modelType in modelTypes

networkInputTypes = ['randomGenerated', 'files']
networkInputType = 'files' # Whether to use random generated input or structured input from a file
taskFile = 'tasks/iris_initial.txt'
interventionsFile = 'tasks/iris_interventions.txt'

if networkInputType == 'randomGenerated':
    numInputs = 32

numHidden = 16
numOutputs = 1
numHiddenLayers = 1
numTotalTasks = 20 # size of initial population, if file given, also applies
numInterventions = 10
autoassociative = False

metric = "mae" # Which metric function to use -- goodness cannot be used for non-binary
                                # input datasets

metricFunction = None






minimumGoodness = 0.95
initialMinimumGoodness = minimumGoodness
dropout = 0.0
minimumMAE = 0.02


numExperiments = 20 # How many runs of the experiment, *not* repeats (need to rename)

stepSize = 2000 # The number of epochs to train before checking & printing
printRate = 2000 # How many steps to train for before printing
maxEpochs = 70000
initialMaxEpochs = 200000 # How many epochs to train for initially

