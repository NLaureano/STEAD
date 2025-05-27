import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time


seismodata_file = "merged/merge.hdf5"
meta_file = "merged/merge.csv"

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
    
label_map = {
    "LOW": 0,
    "MED": 1,
    "HIGH": 2,
    "NOISE": 3
}
# retrieving selected waveforms from the hdf5 file: 
dtfl = h5py.File(seismodata_file, 'r')
#retrieving datasets
#FORMAT: (trace_name : str , index_in_csv : int, LOW/MED/HIGH/NOISE : str)
trainingData = readListFromFile('trainSet.txt')
validationData = readListFromFile('validationSet.txt')
testingData = readListFromFile('testSet.txt')

train_dataset = SeismoDataset(trainingData, dtfl, label_map)
val_dataset = SeismoDataset(validationData, dtfl, label_map)
test_dataset = SeismoDataset(testingData, dtfl, label_map)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)



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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismoCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # weight_decay = L2 regularization

num_epochs = 10    

import TimeKeeper as timeKeeper

totalProgressTimer = timeKeeper.timeKeeper("TOTAL TIMER", num_epochs, 1)
for epoch in range(num_epochs): #Training Loop
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    trainingTimer = timeKeeper.timeKeeper(f"TRAINING EPOCH {epoch}", len(train_loader), 1)
    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)

        waveforms = waveforms.float()  # (N, 3, 6000)
        
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        trainingTimer.timerBroadcast()
    trainingTimer.endTimer()
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

    #Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    validationTimer = timeKeeper.timeKeeper(f"VALIDATION Ep{epoch}", len(val_loader), 1)
    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            waveforms = waveforms.float()
            outputs = model(waveforms)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            validationTimer.timerBroadcast()
    validationTimer.endTimer()
    val_acc = val_correct / val_total

    totalProgressTimer.timerBroadcast()
    print(f"Validation Accuracy: {val_acc:.4f}")
totalProgressTimer.endTimer()
torch.save(model.state_dict(), "seismo_cnn_weights.pth")
# model = SeismoCNN(num_classes=4)  # Recreate the model architecture
# model.load_state_dict(torch.load("seismo_cnn_weights.pth"))
# model.to(device)  # Don't forget this if using CUDA
# model.eval()  # Set to evaluation mode for inference