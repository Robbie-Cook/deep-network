<h1>Deep Network</h1>

An implementation of a tensorflow neural network with pseudorehearsal implemented.
The types of pseudorehearsal implemented with this network are sweep rehearsal and sweep pseudorehearsal.
These are implemented as per *Catastrophic Forgetting, Rehearsal, and Pseudorehearsal* by A. Robins 
(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.108.3078&rep=rep1&type=pdf). 

My experimentation focusses on exploring what happens as the network is expanded, and as deep network techniques are added.

The main file is `main.py`.  `rehearsal.py` contains the rehearsal implementation. `settings.py` contains most of the settings.
