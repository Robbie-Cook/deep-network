# Import libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import subprocess

# Import self-made files
import task
import metrics
import rehearsal
import settings
import myio


"""
Settings
"""

"""
Main routine
"""

summedAverages = np.array([0.0 for i in range(settings.numInterventions+1)])

for i in range(settings.numExperimentRepeats): # repeat entire experiment

    # Make a network
    model = keras.Sequential()

    model.add(keras.layers.Dense(settings.numInputs, activation='sigmoid'))
    for layer in range(settings.numHiddenLayers):
        model.add(keras.layers.Dense(settings.numHidden, activation='sigmoid'))
    model.add(keras.layers.Dense(settings.numOutputs, activation='sigmoid'))

    model.compile(
        optimizer=tf.train.MomentumOptimizer(
            learning_rate=settings.learning_rate,
            momentum=settings.momentum,
            use_nesterov=False,
            
        ),
        loss = 'mse',
        metrics = []
    )

    # Make a bunch of tasks for the network to learn
    mytask = task.createTasks(
        numInputs=settings.numInputs,
        numOutputs=settings.numOutputs,
        numTasks=settings.numTotalTasks
    )

    # Intervening tasks
    interventions = task.createTasks(
        numInputs=settings.numInputs,
        numOutputs=settings.numOutputs,
        numTasks=settings.numInterventions
    )
    
    print("-"*30)
    print("Beginning initial training on base population:")
    task.train(model=model, tasks=mytask, maxEpochs=settings.initialMaxEpochs, minimumGoodness=settings.initialMinimumGoodness)

    goodnesses = rehearsal.rehearse(model=model, method=settings.method, tasks=mytask, interventions=interventions)
    

    for j in range(len(goodnesses)):
        summedAverages[j] += goodnesses[j]
    
    print("\nFinished experiment", i+1)
    print()

# All experiments completed
averagedAverages = [average/(settings.numExperimentRepeats) for average in summedAverages]
print("Finished")
print("Averages:", averagedAverages)

# Write output files
directory = 'data'
outputFileName = myio.get_file_name(directory=directory, name="output")
outputFile = open(directory+"/"+outputFileName, 'w')
[outputFile.write(str(i)+"\n") for i in averagedAverages]

# Write info file
infoFile = open('log/' + outputFileName.split(".")[0] + "_log.txt", 'w')
data = {
    'outputFile': str(outputFile),
    'method': settings.method,
    'numInputs' : settings.numInputs,
    'numHidden' : settings.numHidden,
    'numOutputs' : settings.numOutputs,
    'Base population': settings.numTotalTasks,
    'numInterventions' : settings.numInterventions,
    'numExperimentRepeats': settings.numExperimentRepeats,
}

for i in data.keys():
    infoFile.write("{}: {}\n".format(i,data[i]))
