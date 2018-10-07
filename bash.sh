#!/bin/bash

python3 main.py --method=catastrophicForgetting --numHiddenLayers=2
python3 main.py --method=pseudoSweep --numHiddenLayers=2
python3 main.py --method=pseudoSweep --auxNetwork=True --numHiddenLayers=2
python3 main.py --method=sweep --numHiddenLayers=2

python3 main.py --method=catastrophicForgetting --numHiddenLayers=4
python3 main.py --method=pseudoSweep --numHiddenLayers=4
python3 main.py --method=pseudoSweep --auxNetwork=True --numHiddenLayers=4
python3 main.py --method=sweep --numHiddenLayers=4

python3 main.py --method=catastrophicForgetting --numHiddenLayers=6
python3 main.py --method=pseudoSweep --numHiddenLayers=6
python3 main.py --method=pseudoSweep --auxNetwork=True --numHiddenLayers=6
python3 main.py --method=sweep --numHiddenLayers=6

python3 main.py --method=catastrophicForgetting --numHiddenLayers=8
python3 main.py --method=pseudoSweep --numHiddenLayers=8
python3 main.py --method=pseudoSweep --auxNetwork=True --numHiddenLayers=8
python3 main.py --method=sweep --numHiddenLayers=8

python3 main.py --method=catastrophicForgetting --numHiddenLayers=10
python3 main.py --method=pseudoSweep --numHiddenLayers=10
python3 main.py --method=pseudoSweep --auxNetwork=True --numHiddenLayers=10
python3 main.py --method=sweep --numHiddenLayers=10
