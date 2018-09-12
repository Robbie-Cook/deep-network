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

summedAverages = np.array([0.0 for i in range(settings.numInterventions+1)])

for i in range(settings.numExperimentRepeats): # repeat entire experiment


    # Make the network
    model = network.model

    # Make a bunch of tasks for the network to learn
    # If there is a file given, learn tasks from the file
    mytask = None
    interventions = []

    if settings.dataFile != None:
        print("Using datafile", settings.dataFile, "as input")
        mytask = task.taskFromFile(settings.dataFile)
        assert settings.numInputs == len(mytask[0]['input']), "settings.numInputs must be the length of the inputs"
        assert settings.numOutputs == len(mytask[0]['teacher']), "settings.numOutputs must be the length of the inputs"
    
        # Remove some random tasks from the tasks to be the interventions
        interventions = [] 
        assert settings.numInterventions+settings.numTotalTasks <= len(mytask), \
                        ("Base population+numInterventions too high -- only {} tasks".format(len(mytask)))

        for x in range(settings.numInterventions):
            index = random.randrange(0,len(mytask))
            interventions.append(mytask[index])
            del(mytask[index])

    else: 
        mytask = task.createTasks(
            numInputs=settings.numInputs,
            numOutputs=settings.numOutputs,
            numTasks=settings.numTotalTasks
        )

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
}

for i in data.keys():
    infoFile.write("{}: {}\n".format(i,data[i]))
