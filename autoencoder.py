"""
Autoencoder network for generating inputs for psuedoitems
"""

import tensorflow as tf
from tensorflow import keras
import settings

"""
Set up the autoencoder
"""

m = keras.models.Sequential()
m.add(keras.layers.Dense(settings.numInputs/2))
m.add(keras.layers.Dense(settings.numInputs))
m.compile(loss='mse', optimizer = 'adam', metrics=['mae'])


"""
Train the autoencoder
"""

def train(X):
    maxEpochs = settings.maxEpochs
    m.fit(X, X, epochs=maxEpochs, verbose=0)


"""
Do predictions
"""
def predict(X):
    # print(X)
    return m.predict(X)


    
