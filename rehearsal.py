import metrics
import settings
import task
import numpy as np
import copy
import random
import autoencoder

"""
Rehearsal.py

A module which implements the rehearsal algorithms 
Currently implements psuedoSweep and sweep, catastrophic forgetting
"""

"""
Main function

@param method: the rehearsal method to use (catastrophicForgetting, pseudoSweep, sweep)
returns an array of the goodnesses on the base population
"""
def rehearse(model, method, tasks, interventions):
    methods = ['catastrophicForgetting', 'pseudoSweep', 'sweep']
    assert method in methods, "rehearsal method not supported"


    print("-"*30)
    print("Beginning initial training on base population:")
    print("------------")

    print(tasks)

    print("------------")
    initialEpochs = task.train(model=model, tasks=tasks, maxEpochs=settings.initialMaxEpochs, minimumGoodness=settings.initialMinimumGoodness)

    aux = None
    if settings.auxNetwork and settings.method == 'pseudoSweep':
        items = [x['input'] for x in tasks]
        print("Training Autoencoder")
        autoencoder.train(  np.array(items) ) 


    X = [t['input'] for t in tasks]
    Y = [t['teacher'] for t in tasks]
    initialGoodness = metrics.getAccuracyOnTask(model, tasks)

    print("-"*30)
    print("Starting rehearsal:", method)
    print()

    goodnesses = [initialGoodness]
    epochsArray = [initialEpochs]
    learned = copy.copy(tasks)

    for i,intervention in enumerate(interventions):      
        epochs = None
        if method == methods[0]: # Catastrophic Forgetting
            epochs = task.train(model, [intervention])
            print("Trained intervention to accuracy: ", metrics.getAccuracyOnTask(model, [intervention]))

        elif method == methods[1]:# PseudoSweep
            pseudoItems = [createPsuedoItem(model, settings.ranges) for i in range(128)]
            epochs = sweepTrain(model=model, itemsLearned=pseudoItems, intervention=intervention)

        elif method == methods[2]: # Sweep
            epochs = sweepTrain(model=model, itemsLearned=learned, intervention=intervention)
            learned.append(intervention)
        
        else:
            raise ValueError("Error: incorrect method chosen")

        goodness = metrics.getAccuracyOnTask(model, tasks)
        goodnesses.append(goodness)
        epochsArray.append(epochs)
        print("\nLoss {} after {} round {}: {}".format(settings.metric, method, i+1, goodness))
        print()
    
    return {"goodnesses": goodnesses, 'epochs':epochsArray}
        


"""
Create a "psuedo item" -- a random input linked to generated output

returns:

item {
    'input': [0,0,1,...] # random input
    'teacher: [0.3, 0.04,...] # corresponding output
}

If ranges given e.g. [(0,1), (0,1), (0,1), (0,1)], generates a random value in the range
"""
def createPsuedoItem(model, ranges=None):
    # Generate random numbers
    test = None
    item = {}
    # if files are input, get ranges and make sure those are the chosen items 
    if ranges==None or settings.useRanges == False:
        item = task.createRandomTask()
    else:
        test = []
        for range in ranges:
            test.append(random.uniform(range[0],range[1]))
        item['input'] = test

    # if auxilliary network is to be used to generate the pseudoitems
    if settings.auxNetwork:
        item['input'] = autoencoder.predict( np.array([item['input']]) )[0]

            
    item['teacher'] = model.predict( np.array([item['input']]) )[0]
    return item


"""
Train using sweep rehearsal on input buffer
If the itemsLearned are pseudoItems, then pseudoRehearsal is implemented

(Don't forget to add to learned array after completion)
"""
def sweepTrain(model, itemsLearned, intervention):
    X_intervention = np.array([intervention['input']])
    Y_intervention = np.array([intervention['teacher']])
    epochs = 0

    while not task.isTrained(model=model, tasks=[intervention]):
        if epochs > settings.maxEpochs:
            break
        sweep_run_epoch(model=model, itemsLearned=itemsLearned, intervention=intervention)
        epochs+=settings.bufferRefreshRate

        if epochs % settings.printRate == 0:
            print("Training... Loss on intervention: {}, epochs: {}".format(metrics.getAccuracyOnTask(model, np.array([intervention])), epochs))
    
    print("Finished training... Loss on intervention: {}, epochs: {}".format(metrics.getAccuracyOnTask(model, np.array([intervention])), epochs))
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

