"""
Testing file
"""
import argparse

tester = [
    'method',
    'numHiddenLayers'
]

parser = argparse.ArgumentParser()
# parser.add_argument('--method')
# parser.add_argument('--numHiddenLayers')
[parser.add_argument("--" + key) for key in tester]


args = parser.parse_args()
for i in tester:
    if(args[i] != None):
        print("~~method")