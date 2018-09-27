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

args = parser.parse_args()
if(args.method != None):
    settings.method = args.method

if(args.numHiddenLayers != None):
    settings.method = int(args.numHiddenLayers)

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
    mytask = None
    interventions = []

    if settings.useFiles: # If using datafile
        print("Using datafile", settings.dataFile, "as input")
        mytask = task.tasksFromFile(settings.dataFile)
        interventions = task.tasksFromFile(settings.interventionsDataFile)
        settings.numInterventions = len(interventions)
        settings.numInputs = len(mytask[0]['input'])
        settings.basePopulationSize = len(mytask)

    
        # Get interventions from interventions file
        # interventions = []

        # for x in range(settings.numInterventions):
        #     index = random.randrange(0,len(mytask))
        #     interventions.append(mytask[index])
        #     del(mytask[index])

    else: 
        mytask = task.createTasks(
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
    
    # Make the network
    model = network.get_network()
    
    mydict = rehearsal.rehearse(model=model, method=settings.method, tasks=mytask, interventions=interventions)
    
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


# Write output files
directory = 'data'
outputFileName = myio.get_file_name(directory=directory, name="output")
outputFile = open(directory+"/"+outputFileName, 'w')
[outputFile.write(str(i)+"\n") for i in averagedAverages]

print("Finished -- data saved in", outputFileName)
print("Averages:", averagedAverages)

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
    'Base population': settings.basePopulationSize,
    'numInterventions' : settings.numInterventions,
    'numExperiments': settings.numExperiments,
    'numHiddenLayers': settings.numHiddenLayers,
}

for i in data.keys():
    infoFile.write("{}: {}\n".format(i,data[i]))

