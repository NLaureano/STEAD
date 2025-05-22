import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

file_name = "merged/merge.hdf5"
csv_file = "merged/merge.csv"

def saveList(listToSave, fileToSaveTo):
    with open(fileToSaveTo, 'w') as file:
        for item in listToSave:
            # Write as comma-separated values
            file.write(f"{item[0]},{item[1]},{item[2]}\n")

def readListFromFile(fileToReadFrom):
    readList = list()
    with open(fileToReadFrom, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            traceName, ind, y = line.split(',')
            traceName = traceName.strip()
            indInt = int(ind.strip())
            y = y.strip()
            readList.append((traceName, indInt, y))
    return readList

data_list = readListFromFile('LowEqIndexes.txt')
trace_name, index, label_str = data_list[0]
# Retrieve waveform from HDF5
dtfl = h5py.File(file_name, 'r')
dataset = dtfl.get('data/' + str(trace_name))
data = np.array(dataset)
print(data)