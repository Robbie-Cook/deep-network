"""
Settings.py -- Project settings
"""

import tensorflow as tf


optimizers = {
    "momentum": tf.train.MomentumOptimizer,
    "adam": tf.train.AdamOptimizer
}

"""
Settings
"""

method = 'sweep' # sweep, pseudoSweep, catastrophicForgetting
bufferSize = 4 # Buffer size for sweep and pseudosweep

auxNetwork = False
 

numInputs = 32
numHidden = 16
numOutputs = 32
numHiddenLayers = 8
dataFile = None #'tasks/XOR.txt' # None for random task
classifications = None #[0,1]# `None` for regression, otherwise a list of classes for classifications

reluLayers = False


optimizer = optimizers['momentum']


loss = "mse" # Loss function 
metric = "mae" # Which metric function to use -- goodness cannot be used for non-binary
                                # input datasets
metricFunction = None
minimumGoodness = 0.95
minimumMAE = 0.02
initialMinimumGoodness = minimumGoodness

numTotalTasks = 1 # size of initial population, if file given, also applies
numInterventions = 10
numExperimentRepeats = 10 # How many runs of the experiment, *not* repeats (need to rename)
autoassociative = True


stepSize = 2000 # The number of epochs to train before checking & printing
printRate = 2000 # How many steps to train for before printing
maxEpochs = 100000
initialMaxEpochs = 100000 # How many epochs to train for initially


assert optimizer in [i[-1] for i in optimizers.items()], "Incorrect optimiser"
if optimizer == tf.train.MomentumOptimizer:
    learning_rate = 0.3
    momentum = 0.9
    dropout = 0.0
elif optimizer == tf.train.AdamOptimizer:
    learning_rate=0.001
    beta1=0.9
    beta2=0.999
    epsilon=1e-08
    dropout = 0.0