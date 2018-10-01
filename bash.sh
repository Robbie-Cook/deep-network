#!/bin/bash

python3 main.py --method=pseudoSweep --bufferRefreshRate=100
python3 main.py --method=sweep --bufferRefreshRate=100

python3 main.py --method=pseudoSweep --bufferRefreshRate=1000
python3 main.py --method=sweep --bufferRefreshRate=1000
