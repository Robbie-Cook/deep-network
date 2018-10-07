import random
import copy
import numpy as np
import metrics
import settings
import math

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

If the classifications parameter is a list (e.g. [0,1,2]) and not `None`, then the classes shall be those
given in the list, assigned randomly

@return: Returns a dictionary with input and output e.g. {'input': [0,1,0], 'output': [1,0,0]}
"""
def createRandomTask(numInputs=settings.numInputs, 
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
def createRandomTasks(
                numTasks,
                numInputs=settings.numInputs,
                numOutputs=settings.numOutputs):
    assert numTasks > 0, "Number of tasks must be greater than 0"
    return [createRandomTask(numInputs, numOutputs) for i in range(numTasks)]

"""
Make a new task from a file given

The number of outputs in settings.py determines how many outputs there are
"""
def tasksFromFile(datafile):
    myfile = open(datafile)
    mylist = myfile.readlines()
    mylist = [l.replace("\n","").split() for l in mylist]
    tasks = []
    for i,m in enumerate(mylist):
        task = {}
        task['input'] = list(m[:settings.numInputs])
        task['teacher'] = list(m[settings.numInputs:])
        assert len(task['input']) == settings.numInputs, \
            ("When a file used, numInputs ({}) must match file inputs: {}".format(
                                                                    settings.numInputs, 
                                                                    len(task['input'])))
        # Convert to numbers
        for x in range(len(task['input'])):
            task['input'][x] = float(task['input'][x])
    
        for x in range(len(task['teacher'])):
            task['teacher'][x] = float(task['teacher'][x])

        tasks.append(task)

    return tasks



"""
Make tasks based on an X,Y array given
"""
def makeTaskFromXY(X,Y):
    tasks = []
    for i in range(len(X)):
        tasks.append({'input': X[i], 'teacher': Y[i]})

    return tasks

"""
Determine if the network is trained to criterion,
Returns true if network is trained to criterion, false otherwise

"""
def isTrained(model, tasks):

    met = metrics.getAccuracyOnTask(model=model, tasks=tasks)
    if settings.metric == 'goodness':
        # Goodness
        return met > settings.minimumGoodness
    elif settings.metric == 'accuracy':
        # Accuracy
        return met > settings.minimumAccuracy
    elif settings.metric == 'mae':
        # MAE
        return met < settings.minimumMAE
    else:
        raise ValueError('Err: Metric not recognised by isTrained()')

"""
Train a task in the network

@param metricsTask -- if given, train until this task reaches a certain goodness

"""
def train(model, 
          tasks,
          maxEpochs=settings.maxEpochs,
          minimumGoodness=settings.minimumGoodness):

    epochs = 0
    X = np.array([x['input'] for x in tasks])
    Y = np.array([y['teacher'] for y in tasks])

    while(not isTrained(model=model, tasks=tasks) and epochs < maxEpochs):
        for _ in range(settings.stepSize):
            model.train_on_batch(X,Y)

        epochs+= settings.stepSize

        if epochs % (settings.printRate) == 0:
            print("Training task.... Metric value: {}, Epochs: {}/{}".format(
                metrics.getAccuracyOnTask(model=model, tasks=tasks), epochs, maxEpochs))

        

    print("Finished training!.... Loss: {}, Epochs: {}/{}".format(
        metrics.getAccuracyOnTask(model=model, tasks=tasks), epochs, maxEpochs))
    return epochs

