import os
"""
myio.py

Does file-storing overhead, like making new, unique, files etc.
""" 


"""
Makes a new unique file and returns it

Takes: a file directory and then name to save to e.g. 'data'
If this file is taken, the file is saved as output1.txt, etc.
"""
def get_file_name(directory, name):
    number = 0
    suffix = '.txt'
    myfile = name+suffix

    while myfile in os.listdir(directory):
        myfile = name + str(number) + suffix
        number += 1

    return myfile
