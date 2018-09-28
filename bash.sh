#!/bin/bash

python3 main.py --method=pseudoSweep --bufferRefreshRate=1
python3 main.py --method=sweep --bufferRefreshRate=1

python3 main.py --method=pseudoSweep --bufferRefreshRate=4
python3 main.py --method=sweep --bufferRefreshRate=4

python3 main.py --method=pseudoSweep --bufferRefreshRate=8
python3 main.py --method=sweep --bufferRefreshRate=8

python3 main.py --method=pseudoSweep --bufferRefreshRate=10
python3 main.py --method=sweep --bufferRefreshRate=10

python3 main.py --method=pseudoSweep --bufferRefreshRate=20
python3 main.py --method=sweep --bufferRefreshRate=20

python3 main.py --method=pseudoSweep --bufferRefreshRate=50
python3 main.py --method=sweep --bufferRefreshRate=50

python3 main.py --method=pseudoSweep --bufferRefreshRate=100
python3 main.py --method=sweep --bufferRefreshRate=100

python3 main.py --method=pseudoSweep --bufferRefreshRate=1000
python3 main.py --method=sweep --bufferRefreshRate=1000
