import pandas as pd
import h5py
import SeismoCNNImport
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
model = SeismoCNNImport.SeismoCNN(num_classes=4)  # Recreate the model architecture
model.load_state_dict(torch.load("seismo_cnn_weights.pth"))
model.to(device)  
model.eval()  # Set to evaluation mode for inference

seismodata_file = "merged/merge.hdf5"
meta_file = "merged/merge.csv"

dtfl = h5py.File(seismodata_file, 'r')
testingData = readListFromFile('testSet.txt')
test_dataset = SeismoCNNImport.SeismoDataset(testingData, dtfl, label_map)
test_loader = DataLoader(test_dataset, batch_size=32)

#Testing phase
model.eval()
val_correct = 0
val_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for waveforms, labels in test_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        waveforms = waveforms.float()
        outputs = model(waveforms)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
val_acc = val_correct / val_total
print(f"Validation Accuracy: {val_acc:.4f}")

score = np.zeros((5, 5))
for i in range(len(all_labels)):
    score[all_preds[i]][all_labels[i]] += 1
    score[all_preds[i]][4] += 1
    score[4][all_labels[i]] += 1

# Print confusion matrix with labels
labels = ["LOW", "MED", "HIGH", "NOISE", "Total"]
print("\nConfusion Matrix (Predicted \\ Actual):")
header = "{:>8}".format("")
for label in labels:
    header += "{:>8}".format(label)
print(header)
for i, row_label in enumerate(labels):
    row = "{:>8}".format(row_label)
    for j in range(5):
        row += "{:8.0f}".format(score[i][j])
    print(row)
