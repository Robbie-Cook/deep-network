"""
Settings.py -- Project settings
"""

method = 'pseudoSweep'
bufferSize = 4 # Buffer size for sweep and pseudosweep

auxNetwork = False

numInputs = 32
numHidden = 16
numOutputs = 32
numHiddenLayers = 1
dataFile = None #'tasks/XOR.txt' # None for random task
classifications = None #[0,1]# `None` for regression, otherwise a list of classes for classifications

reluLayers = False

learning_rate = 0.3
momentum = 0.5
dropout = 0
loss = "mse" # Loss function 
metric = "mae" # Which metric function to use -- goodness cannot be used for non-binary
                                # input datasets
metricFunction = None

numTotalTasks = 5 # size of initial population
numInterventions = 10
numExperimentRepeats = 10 # How many runs of the experiment, *not* repeats (need to rename)
autoassociative = False

minimumGoodness = 0.95
minimumMAE = 0.02
initialMinimumGoodness = minimumGoodness

stepSize = 2000 # The number of epochs to train before checking & printing
printRate = 2000 # How many steps to train for before printing
maxEpochs = 70000
initialMaxEpochs = 200000

