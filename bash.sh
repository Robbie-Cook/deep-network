#!/bin/bash

python3 main.py --method=catastrophicForgetting --dropout=0.0
python3 main.py --method=pseudoSweep --dropout=0.0
python3 main.py --method=sweep --dropout=0.0

python3 main.py --method=catastrophicForgetting --dropout=0.1
python3 main.py --method=pseudoSweep --dropout=0.1
python3 main.py --method=sweep --dropout=0.1

python3 main.py --method=catastrophicForgetting --dropout=0.2
python3 main.py --method=pseudoSweep --dropout=0.2
python3 main.py --method=sweep --dropout=0.2

python3 main.py --method=catastrophicForgetting --dropout=0.3
python3 main.py --method=pseudoSweep --dropout=0.3
python3 main.py --method=sweep --dropout=0.3

python3 main.py --method=catastrophicForgetting --dropout=0.4
python3 main.py --method=pseudoSweep --dropout=0.4
python3 main.py --method=sweep --dropout=0.4

python3 main.py --method=catastrophicForgetting --dropout=0.5
python3 main.py --method=pseudoSweep --dropout=0.5
python3 main.py --method=sweep --dropout=0.5