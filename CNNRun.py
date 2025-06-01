import pandas as pd
import h5py
import SeismoCNNImport as SeismoCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from SeismoCNNImport import readListFromFile
import numpy as np
import TimeKeeper as timeKeeper

label_map = {
    "LOW": 0,
    "MED": 1,
    "HIGH": 2,
    "NOISE": 3
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismoCNN(num_classes=4)  # Recreate the model architecture
model.load_state_dict(torch.load("seismo_cnn_weights.pth"))
model.to(device)  
model.eval()  # Set to evaluation mode for inference

seismodata_file = "merged/merge.hdf5"
meta_file = "merged/merge.csv"

dtfl = h5py.File(seismodata_file, 'r')
testingData = readListFromFile('testSet.txt')