# Import libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse


# Import self-made files
import task
import metrics
import rehearsal
import settings
import myio


"""
Parse arguments
"""

parser = argparse.ArgumentParser()
parser.add_argument('--method')
parser.add_argument('--numHiddenLayers')

args = parser.parse_args()
if(args.method != None):
    settings.method = args.method

if(args.numHiddenLayers != None):
    settings.numHiddenLayers = int(args.numHiddenLayers)


"""
Main routine
"""

summedAverages = np.array([0.0 for i in range(settings.numInterventions+1)])

for i in range(settings.numExperimentRepeats): # repeat entire experiment

    # Implement the auxilliary network if needed
    if settings.auxNetwork:
        print("Caution: using auxilliary network")

        aux = keras.Sequential()
        # aux.add(keras.layers.Dense())




    # Make the main network
    model = keras.Sequential()

    model.add(keras.layers.Dense(settings.numInputs, activation='sigmoid'))
    for layer in range(settings.numHiddenLayers):
        model.add(keras.layers.Dropout(settings.dropout)) # Dropout
        model.add(keras.layers.Dense(settings.numHidden, activation=('sigmoid' if not settings.reluLayers else 'relu')))
    model.add(keras.layers.Dense(settings.numOutputs, activation='sigmoid'))

    model.compile(
        tf.train.MomentumOptimizer(
            learning_rate=settings.learning_rate,
            momentum=settings.momentum,
            use_nesterov=False,
        ),
        loss = settings.loss,
        metrics = []
    )

    # Make a bunch of tasks for the network to learn
    mytask = task.createTasks(
        numInputs=settings.numInputs,
        numOutputs=settings.numOutputs,
        numTasks=settings.numTotalTasks
    )
    if settings.dataFile != None:
        print("Caution: using datafile", settings.dataFile)
        mytask = task.taskFromFile(settings.dataFile)

    # print("Task", mytask)

    # Intervening tasks
    if (settings.numInterventions > 0):
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
    'numHiddenLayers': settings.numHiddenLayers,
    'reluLayers': settings.reluLayers
}

for i in data.keys():
    infoFile.write("{}: {}\n".format(i,data[i]))
