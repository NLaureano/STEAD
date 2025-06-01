import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

class SeismoDataset(Dataset):
    def __init__(self, data_list, hdf5_file, label_map):
        self.data_list = data_list
        self.hdf5_file = hdf5_file
        self.label_map = label_map

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        trace_name, index, label_str = self.data_list[idx]

        # Retrieve waveform from HDF5
        dataset = self.hdf5_file.get('data/' + str(trace_name))
        waveform = np.array(dataset)  # (6000, 3) with [E, N, Z] columns

        waveform = waveform[:, [2, 1, 0]]  # → still (6000, 3)

        # Convert to torch tensor
        waveform = torch.tensor(waveform, dtype=torch.float32).permute(1, 0) # → (3, 6000)

        # Convert label string to int
        label = self.label_map[label_str]

        return waveform, label


class SeismoCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SeismoCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)  # → length: 3000

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)  # → length: 1500

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)  # → length: 750

        self.fc1 = nn.Linear(64 * 750, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 16, 3000)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 32, 1500)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (B, 64, 750)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)