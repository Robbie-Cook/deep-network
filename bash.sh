#!/bin/bash

python3 main.py --method=catastrophicForgetting --numHiddenLayers=10
python3 main.py --method=pseudoSweep --numHiddenLayers=10
python3 main.py --method=sweep --numHiddenLayers=10

python3 main.py --method=catastrophicForgetting --numHiddenLayers=5
python3 main.py --method=pseudoSweep --numHiddenLayers=5
python3 main.py --method=sweep --numHiddenLayers=5
