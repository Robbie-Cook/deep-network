"""
Plot.py

Plot an output file
"""

import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import settings

assert len(sys.argv) > 1, "Please enter a file to plot"

values = open(sys.argv[1], 'r').read().split()
values = [float(v) for v in values]

# Data for plotting
t = np.arange(0, settings.numInterventions+1, 1)
s = values

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='Num Interventions', ylabel='Mean Goodness',
       title=sys.argv[1])
ax.grid()


myfile = sys.argv[1].split("/")[-1]
fig.savefig("graphs/" + myfile +".png")
plt.ylim((0,1.0))
plt.show()
