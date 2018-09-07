import random
import copy
import numpy as np
import metrics
import settings

"""
Task.py

Generate an input/output pair for the network to use

e.g. [0,1,0,1] -> [1,0,1,0]
"""

"""
Create a new task for the neural network
i.e. a mapping of inputs to outputs

@param autoassociative: whether the input maps to itself, or whether it maps heteroassociatively
e.g. autoassociative: [1,0] -> [1,0]
     heteroassociative: [1,0] -> [0,0]

@return: Returns a dictionary with input and output e.g. {'input': [0,1,0], 'output': [1,0,0]}
"""
def createTask(numInputs=settings.numInputs, 
                numOutputs=settings.numOutputs):
    task = {}
    task['input'] = np.array([random.randrange(0,2) for i in range(numInputs)])

    if settings.autoassociative:
        task['teacher'] = copy.copy(task['input'])
    else:
         task['teacher'] = np.array([random.randrange(0,2) for i in range(numOutputs)])

    return task


"""
Create a set of tasks using createTask()
"""
def createTasks(
                numTasks,
                numInputs=settings.numInputs,
                 numOutputs=settings.numOutputs):
    assert numTasks > 0, "Number of tasks must be greater than 0"
    return [createTask(numInputs, numOutputs) for i in range(numTasks)]

"""
Train a task in the network

@param metricsTask -- if given, train until this task reaches a certain goodness

"""
def train(model, 
          tasks, 
          metricsTask=None, 
          maxEpochs=settings.maxEpochs,
          minimumGoodness=settings.minimumGoodness):
    X = np.array([tasks[i]['input'] for i in range(len(tasks))]) # Inputs
    Y = np.array([tasks[i]['teacher'] for i in range(len(tasks))]) # Teaching outputs

    epochs = 0
    goodness = metrics.getGoodness(model.predict(X), Y) 
    
    x_metric, y_metric = None, None
    if metricsTask: # get goodness from the metric task
        x_metric = np.array([metricsTask['input']])
        y_metric = np.array([metricsTask['teacher']])
        goodness = metrics.getGoodness(model.predict(x_metric), y_metric)
        
    while(goodness < minimumGoodness and epochs < maxEpochs):
        for i in range(settings.stepSize):
            model.train_on_batch(X,Y)
        
        if metricsTask != None:
            goodness = metrics.getGoodness(model.predict(x_metric), y_metric)
        else:
            goodness = metrics.getGoodness(model.predict(X), Y)

        epochs+= settings.stepSize

        if epochs % (settings.printRate) == 0:
            print("Training task.... Goodness: {}, Epochs: {}/{}".format(goodness, epochs, maxEpochs))
    print("Finished training!.... Goodness: {}, Epochs: {}/{}".format(goodness, epochs, maxEpochs))