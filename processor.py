import sys

"""
Processor.py

Puts the information into a form which can be put directly into a LaTeX pgfplots graph.
"""

file = open(sys.argv[1]).read().split()


for i,item in enumerate(file):
    print("({}, {})".format(i,item))
