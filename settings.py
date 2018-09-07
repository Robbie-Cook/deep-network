"""
Settings.py -- Project settings
"""

method = 'sweep'

numInputs = 32
numHidden = 16
numOutputs = 32
numHiddenLayers = 1
maxEpochs = 50000
initialMaxEpochs = 100000


learning_rate = 0.3
momentum = 0.9

numTotalTasks = 20 # size of initial population
numInterventions = 10
numExperimentRepeats = 10
autoassociative = False

minimumGoodness = 0.95
initialMinimumGoodness = minimumGoodness

stepSize = 2000 # The number of epochs to train before checking & printing
printRate = 10000 # How many steps to train for before printing