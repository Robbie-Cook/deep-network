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
def createTask(numInputs=settings.numInputs, 
                numOutputs=settings.numOutputs,
                classifications=settings.classifications):
    task = {}
    task['input'] = np.array([random.randrange(0,2) for i in range(numInputs)])

    if classifications == None: # Generate a teacher for regression (whole number, within the classes declared
        if settings.autoassociative:
            task['teacher'] = copy.copy(task['input'])
        else:
            task['teacher'] = np.array([random.randrange(0,2) for i in range(numOutputs)])

    else: # Generate a teacher for classification
        assert (len(classifications) == 1 and numOutputs == 1) \
            or math.ceil(math.log10(len(classifications))) == numOutputs , \
            "Number of outputs must be correct for classification"
        task['teacher'] = np.array([random.randrange(0,len(classifications))])

    return task

"""
Make a new task from a file given
"""
def taskFromFile(datafile):
    myfile = open(datafile)
    mylist = myfile.readlines()
    mylist = [l.replace("\n","").split() for l in mylist]
    assert len(mylist[0]) == settings.numInputs + settings.numOutputs, "The number of outputs and inputs do "
    tasks = []
    for i,m in enumerate(mylist):
        task = {}
        task['input'] = list(m[:settings.numInputs])
        task['teacher'] = list(m[settings.numInputs:])

        # Convert to numbers
        for x in range(len(task['input'])):
            task['input'][x] = float(task['input'][x])
    
        for x in range(len(task['teacher'])):
            task['teacher'][x] = float(task['teacher'][x])
        
        tasks.append(task)


    print(tasks)

    # for i in len('input')
    return tasks


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
Make tasks based on an X,Y array given
"""
def makeTaskFromXY(X,Y):
    tasks = []
    for i in range(len(X)):
        tasks.append({'input': X[i], 'teacher': Y[i]})

    return tasks


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
    loss = settings.metricFunction(model.predict(X), Y) 
    
    x_metric, y_metric = None, None
    if metricsTask: # get goodness from the metric task
        x_metric = np.array([metricsTask['input']])
        y_metric = np.array([metricsTask['teacher']])
        loss = settings.metricFunction(model.predict(x_metric), y_metric)
    
    assert settings.metricFunction in [metrics.getGoodness, metrics.getMAE], \
            "Metric function must be defined"
    if settings.metricFunction == metrics.getGoodness: # If goodness is the metric    
        while(loss < minimumGoodness and epochs < maxEpochs):
            for i in range(settings.stepSize):
                model.train_on_batch(X,Y)
            
            if metricsTask != None:
                loss = settings.metricFunction(model.predict(x_metric), y_metric)
            else:
                loss = settings.metricFunction(model.predict(X), Y)

            epochs+= settings.stepSize

            if epochs % (settings.printRate) == 0:
                print("Training task.... Goodness: {}, Epochs: {}/{}".format(loss, epochs, maxEpochs))

    elif settings.metricFunction == metrics.getMAE:
        while(loss > settings.minimumMAE and epochs < maxEpochs):
            for i in range(settings.stepSize):
                model.train_on_batch(X,Y)
            
            if metricsTask != None:
                loss = settings.metricFunction(model.predict(x_metric), y_metric)
            else:
                loss = settings.metricFunction(model.predict(X), Y)

            epochs+= settings.stepSize

            if epochs % (settings.printRate) == 0:
                print("Training task.... MAE: {}, Epochs: {}/{}".format(loss, epochs, maxEpochs))
    
    print("Finished training!.... Loss: {}, Epochs: {}/{}".format(loss, epochs, maxEpochs))
