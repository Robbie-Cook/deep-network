"""
Settings.py -- Project settings
"""

import metrics

method = 'sweep'
bufferSize = 4 # Buffer size for sweep and pseudosweep

auxNetwork = True

numInputs = 32
numHidden = 16
numOutputs = 32
numHiddenLayers = 10
dataFile = None #'tasks/XOR.txt' # None for random task
classifications = None #[0,1]# `None` for regression, otherwise a list of classes for classifications

reluLayers = False

learning_rate = 0.3
momentum = 0.5
loss = "mse"
metricFunction = metrics.getMAE # Which metric function to use -- goodness cannot be used for non-binary
                                # input datasets

numTotalTasks = 20 # size of initial population
numInterventions = 10
numExperimentRepeats = 1 # How many runs of the experiment, *not* repeats (need to rename)
autoassociative = False

minimumGoodness = 0.95
initialMinimumGoodness = minimumGoodness

stepSize = 2000 # The number of epochs to train before checking & printing
printRate = 2000 # How many steps to train for before printing
maxEpochs = 50000
initialMaxEpochs = maxEpochs

