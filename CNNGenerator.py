import pandas as pd
import h5py
import SeismoCNNImport as SeismoCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from SeismoCNNImport import readListFromFile
import TimeKeeper as timeKeeper
import time


seismodata_file = "merged/merge.hdf5"
meta_file = "merged/merge.csv"


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
#testingData = readListFromFile('testSet.txt')

train_dataset = SeismoCNN.SeismoDataset(trainingData, dtfl, label_map)
val_dataset = SeismoCNN.SeismoDataset(validationData, dtfl, label_map)
#test_dataset = SeismoCNN.SeismoDataset(testingData, dtfl, label_map)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
#test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismoCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # weight_decay = L2 regularization

num_epochs = 10    

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