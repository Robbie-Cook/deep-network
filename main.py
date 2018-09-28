# Import libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import random


# Import self-made files
import task
import metrics
import rehearsal
import settings
import myio
import main_network
import cnn_network


"""
Which network to use -- important
"""
network = main_network

"""
Parse arguments
"""

parser = argparse.ArgumentParser()
parser.add_argument('--method')
parser.add_argument('--numHiddenLayers')
parser.add_argument('--dropout')

args = parser.parse_args()
if(args.method != None):
    settings.method = args.method

if(args.numHiddenLayers != None):
    settings.numHiddenLayers = int(args.numHiddenLayers)

if(args.dropout != None):
    settings.dropout = float(args.dropout)
    print(settings.dropout)

"""
Main routine
"""

rawData = [] # the output of each epoch for the standard deviation
rawEpochs = []

summedAverages = np.array([0.0 for i in range(settings.numInterventions+1)])


print("Beginning data collection. Method: ", settings.method)

for i in range(settings.numExperiments): # repeat entire experiment

    # Make a bunch of tasks for the network to learn
    # If there is a file given, learn tasks from the file
    mytasks = None
    interventions = []

    if settings.networkInputType == 'files': # If using datafile as input
        print("Using datafile", settings.taskFile, "as input")
        mytasks = task.tasksFromFile(settings.taskFile)
        interventions = task.tasksFromFile(settings.interventionsFile)

        # If model type is classification, convert the intervention tasks and tasks to
        # one-hot vectors
        if settings.modelType == 'classification':
            assert settings.numOutputs == settings.numClasses, "For classification, the num outputs must be the number of classes"

            for intervention in interventions:
                intervention['teacher'] = keras.utils.to_categorical(intervention['teacher'], num_classes=settings.numClasses)[0]
            for t in mytasks:
                t['teacher'] = keras.utils.to_categorical(t['teacher'], num_classes=settings.numClasses)[0]
        # Set number of inputs to be number of features in the given dataset
        settings.numInputs = len(mytasks[0]['input'])

        # Select a random set of interventions from the intervention file
        # which is the size given in settings.py
        random.shuffle(interventions)
        interventions = interventions[:settings.numInterventions]

        # Select a random base population from the input file
        # which is the size given in settings.py
        random.shuffle(mytasks)
        mytasks = mytasks[:settings.basePopulationSize]

        print("Datafile tasks", mytasks, "interventions", interventions)
    
    elif settings.networkInputType == 'randomGenerated': # Otherwise, random generated input used 
        mytasks = task.createTasks(
            numInputs=settings.numInputs,
            numOutputs=settings.numOutputs,
            numTasks=settings.basePopulationSize
        )

        # Intervening tasks
        if (settings.numInterventions > 0):
            interventions = task.createTasks(
                numInputs=settings.numInputs,
                numOutputs=settings.numOutputs,
                numTasks=settings.numInterventions
            )
    else:
        raise ValueError('Incorrect networkInputType')
    
    # Make the network
    model = network.get_network()
    
    mydict = rehearsal.rehearse(model=model, method=settings.method, tasks=mytasks, interventions=interventions)
    
    goodnesses = mydict['goodnesses']
    epochsCount = mydict['epochs']


    for j in range(len(goodnesses)):
        summedAverages[j] += goodnesses[j]
    

    rawData.append(goodnesses) # append individual results for standard deviation
    rawEpochs.append(epochsCount)

    print("\nFinished experiment", i+1)
    print()

# All experiments completed
averagedAverages = [average/(settings.numExperiments) for average in summedAverages]

print("Finished")
print("Averages:", averagedAverages)

# Write output files
directory = 'data'
outputFileName = myio.get_file_name(directory=directory, name="output")
outputFile = open(directory+"/"+outputFileName, 'w')
[outputFile.write(str(i)+"\n") for i in averagedAverages]

## Write the number of epochs taken
epochsFile = open('epochCounts/' + outputFileName.split(".")[0] + "_count.txt", 'w')
for x in rawEpochs:
    for column in x:
        epochsFile.write(str(column) + " ")
    epochsFile.write("\n")

# Write raw data (results for each experiment)
raw_data_file = open('raw_data/' + outputFileName.split(".")[0] + "_raw.txt", 'w')
for x in rawData:
    for column in x:
        raw_data_file.write(str(column) + " ")
    raw_data_file.write("\n")

# Write info file
infoFile = open('log/' + outputFileName.split(".")[0] + "_log.txt", 'w')
data = {
    'outputFile': str(outputFile),
    'method': settings.method,
    'numInputs' : settings.numInputs,
    'numHidden' : settings.numHidden,
    'numOutputs' : settings.numOutputs,
    'base population': settings.basePopulationSize,
    'numInterventions' : settings.numInterventions,
    'numExperiments': settings.numExperiments,
    'numHiddenLayers': settings.numHiddenLayers,
    'dropout': settings.dropout,
    'inputMethod': settings.networkInputType
}

for i in data.keys():
    infoFile.write("{}: {}\n".format(i,data[i]))

