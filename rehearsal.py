import metrics
import settings
import task
import numpy as np
import copy
import random

"""
Rehearsal.py

A module which implements the rehearsal algorithms 
Currently implements psuedoSweep and sweep, catastrophic forgetting
"""

"""
Main function

@param method: the rehearsal method to use (catastrophicForgetting, pseudoSweep, sweep)
returns an array of the goodnesses

InitialEpochs -- how many epochs the network has already been trained for
"""
def rehearse(model, method, tasks, interventions, initialEpochs):
    methods = ['catastrophicForgetting', 'pseudoSweep', 'sweep']
    assert method in methods, "rehearsal method not supported"

    X = [t['input'] for t in tasks]
    Y = [t['teacher'] for t in tasks]
    initialGoodness = metrics.getTaskQuality(model, tasks)

    print("-"*30)
    print("Starting rehearsal:", method)
    print()
    print("Initial loss {}: {}".format(settings.metric, initialGoodness))
    print()

    goodnesses = [initialGoodness]
    epochsArray = [initialEpochs]
    learned = copy.copy(tasks)

    for i,intervention in enumerate(interventions):      
        epochs = None
        if method == methods[0]: # Catastrophic Forgetting
            epochs = task.train(model, [intervention])

        elif method == methods[1]:# PseudoSweep
            pseudoItems = [createPsuedoItem(model) for i in range(128)]
            epochs = sweepTrain(model=model, itemsLearned=pseudoItems, intervention=intervention)

        elif method == methods[2]: # Sweep
            epochs = sweepTrain(model=model, itemsLearned=learned, intervention=intervention)
            learned.append(intervention)
        
        else:
            raise ValueError("Error: incorrect method chosen")

        goodness = metrics.getTaskQuality(model, tasks)
        goodnesses.append(goodness)
        epochsArray.append(epochs)
        print("\nLoss {} after {} round {}: {}".format(settings.metric, method, i+1, goodness))
        print()
    
    if settings.countEpochs:
        return epochsArray
    return goodnesses
        


"""
Create a "psuedo item" -- a random input linked to generated output

returns:

item {
    'input': [0,0,1,...] # random input
    'teacher: [0.3, 0.04,...] # corresponding output
}
"""
def createPsuedoItem(model):
    # Generate random numbers  
    item = task.createTask()
    test = np.array([item['input']])
    item['teacher'] = model.predict(test)[0]
    return item


"""
Train using sweep rehearsal on input buffer
If the itemsLearned are pseudoItems, then pseudoRehearsal is implemented

(Don't forget to add to learned array after completion)
"""
def sweepTrain(model, itemsLearned, intervention):
    X_intervention = np.array([intervention['input']])
    Y_intervention = np.array([intervention['teacher']])
    loss = settings.metricFunction(model.predict(X_intervention),Y_intervention)
    epochs = 0
    if settings.metricFunction == metrics.getGoodness: # goodness

        while loss < settings.minimumGoodness and epochs < settings.maxEpochs:
            sweep_run_epoch(model=model, itemsLearned=itemsLearned, intervention=intervention)
            epochs+=settings.bufferRefreshRate

            if epochs % settings.stepSize == 0:
                loss = settings.metricFunction(model.predict(X_intervention),Y_intervention)
            if epochs % settings.printRate == 0:
                print("Training... Loss on intervention: {}, epochs: {}".format(loss, epochs))
    else: 
        while loss > settings.minimumMAE and epochs < settings.maxEpochs: # MAE
            sweep_run_epoch(model=model, itemsLearned=itemsLearned, intervention=intervention)
            epochs+=settings.bufferRefreshRate

            if epochs % settings.stepSize == 0:
                loss = settings.metricFunction(model.predict(X_intervention),Y_intervention)
            if epochs % settings.printRate == 0:
                print("Training... Loss on intervention: {}, epochs: {}".format(loss, epochs))
    return epochs

"""
The inner function for sweepTrain
"""
def sweep_run_epoch(model, itemsLearned, intervention):
    indices = [random.randrange(0,len(itemsLearned)) for i in range(settings.bufferSize-1)]
    buffer = [itemsLearned[i] for i in indices]
    buffer.append(intervention)
    random.shuffle(buffer)

    X = np.array([b['input'] for b in buffer])
    Y = np.array([b['teacher'] for b in buffer])

    # Train for buffer refresh rate
    # Make sure buffer is learned before moving on
    for i in range(settings.bufferRefreshRate):
        model.train_on_batch(X,Y)

