#!/bin/bash


python3 main.py --method=catastrophicForgetting --populationSize=1 --numHiddenLayers=1
python3 main.py --method=pseudoSweep --populationSize=1 --numHiddenLayers=1
python3 main.py --method=sweep --populationSize=1 --numHiddenLayers=1

python3 main.py --method=catastrophicForgetting --populationSize=5 --numHiddenLayers=1
python3 main.py --method=pseudoSweep --populationSize=5 --numHiddenLayers=1
python3 main.py --method=sweep --populationSize=5 --numHiddenLayers=1

python3 main.py --method=catastrophicForgetting --populationSize=20 --numHiddenLayers=1
python3 main.py --method=pseudoSweep --populationSize=20 --numHiddenLayers=1
python3 main.py --method=sweep --populationSize=20 --numHiddenLayers=1



python3 main.py --method=catastrophicForgetting --populationSize=1 --numHiddenLayers=2
python3 main.py --method=pseudoSweep --populationSize=1 --numHiddenLayers=2
python3 main.py --method=sweep --populationSize=1 --numHiddenLayers=2

python3 main.py --method=catastrophicForgetting --populationSize=5 --numHiddenLayers=2
python3 main.py --method=pseudoSweep --populationSize=5 --numHiddenLayers=2
python3 main.py --method=sweep --populationSize=5 --numHiddenLayers=2

python3 main.py --method=catastrophicForgetting --populationSize=20 --numHiddenLayers=2
python3 main.py --method=pseudoSweep --populationSize=20 --numHiddenLayers=2
python3 main.py --method=sweep --populationSize=20 --numHiddenLayers=2
