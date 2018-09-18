import sys
import numpy as np

"""
Processor.py

Puts the information into a form which can be put directly into a LaTeX pgfplots graph.
"""

myfile = open(sys.argv[1])
raw_lines = [line.split() for line in myfile.readlines()]

if len(raw_lines[0]) > 1: # if the data is raw i.e. isn't average
    raw_lines = np.transpose(raw_lines)

for i,item in enumerate(raw_lines): 
    floatItems = [float(i) for i in item]
    average = sum(floatItems)/len(floatItems)
    stddev = np.std(floatItems)
    print("({}, {}) +- (0, {})".format(i,average,stddev))
