import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SeismoCNNImport as SeismoCNN

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

train_dataset = SeismoCNN.SeismoDataset(trainingData, dtfl, label_map)
val_dataset = SeismoCNN.SeismoDataset(validationData, dtfl, label_map)
test_dataset = SeismoCNN.SeismoDataset(testingData, dtfl, label_map)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import TimeKeeper as timeKeeper

def testHyperparameters(hyperparameter_in_question, hp_range):
    #Defaults:
    learningRate = 0.001
    weightDecay = 1e-5 
    num_epochs = 10 #TODO MAKE THIS 10   --------------------------------------------------------------------------  
    bestHP = 0 #Default the first one is the best
    bestOfTheBestValAcc = 0
    TLosses = []
    TAccs = []
    VLosses = []
    VAccs = []

    hyperparameterTimer = timeKeeper.timeKeeper(f"HP TIMER {hyperparameter_in_question}", len(hp_range), 1)
    for hp_index in range(len(hp_range)): #Iterate through hyperparameters
        trainingLosses = []
        trainingAccuracy = []
        validationLosses = []
        validationAccuracy = []
        bestValAcc = 0

        if hyperparameter_in_question == "lr":
            learningRate = hp_range[hp_index]
        elif hyperparameter_in_question == "wd":
            weightDecay = hp_range[hp_index]
        elif hyperparameter_in_question == "ep":
            num_epochs = hp_range[hp_index]
        else:
            print("ERROR - UNIMPLEMENTED HYPERPARAMETER")
            return 
        
        model = SeismoCNN(num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)  # weight_decay = L2 regularization

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
            trainingLosses.append(running_loss)
            trainingAccuracy.append(train_acc)

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss_total = 0.0 
            validationTimer = timeKeeper.timeKeeper(f"VALIDATION Ep{epoch}", len(val_loader), 1)

            with torch.no_grad():
                for waveforms, labels in val_loader:
                    waveforms, labels = waveforms.to(device), labels.to(device)
                    waveforms = waveforms.float()
                    outputs = model(waveforms)

                    # Compute loss on validation batch
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    validationTimer.timerBroadcast()

            validationTimer.endTimer()

            # Compute average loss and accuracy
            val_acc = val_correct / val_total
            avg_val_loss = val_loss_total / len(val_loader)  # Average over batches

            print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {avg_val_loss:.4f}")
            validationLosses.append(avg_val_loss)
            validationAccuracy.append(val_acc)
            if val_acc > bestValAcc: bestValAcc = val_acc #Update best valAcc
        if bestValAcc > bestOfTheBestValAcc: 
            bestOfTheBestValAcc = bestValAcc
            bestHP = hp_index
        
        #Keep track of each hyperparameter iteration's datasheets
        TLosses.append(trainingLosses)
        TAccs.append(trainingAccuracy)
        VLosses.append(validationLosses)
        VAccs.append(validationAccuracy)

        hyperparameterTimer.timerBroadcast()

    return hp_range[bestHP], TLosses, TAccs, VLosses, VAccs
    
totalProgressTimer = timeKeeper.timeKeeper("TOTAL TUNING TIMER", 2, 1)
learning_rates = [1e-4, 5e-4, 1e-3, 3e-3] #Default = 1e-4
weight_decay_rates = [0, 1e-6, 1e-4, 1e-3] #Default = 1e-5
# Test learning rates and save results
best_lr, lrTrainingLosses, lrTrainingAcc, lrValidationLosses, lrValidationAcc = testHyperparameters("lr", learning_rates)
with open("best_results.txt", "w") as f:
    f.write(f"Best Learning Rate: {best_lr}\n")
    f.write("Training Losses:\n")
    f.write(str(lrTrainingLosses) + "\n")
    f.write("Training Accuracies:\n")
    f.write(str(lrTrainingAcc) + "\n")
    f.write("Validation Losses:\n")
    f.write(str(lrValidationLosses) + "\n")
    f.write("Validation Accuracies:\n")
    f.write(str(lrValidationAcc) + "\n")

# Save learning rate results in a structured format (npz)
np.savez(
    "best_lr_results.npz",
    best_lr=best_lr,
    lrTrainingLosses=lrTrainingLosses,
    lrTrainingAcc=lrTrainingAcc,
    lrValidationLosses=lrValidationLosses,
    lrValidationAcc=lrValidationAcc,
    learning_rates=learning_rates
)

totalProgressTimer.timerBroadcast()

# Test weight decay and save results (append to file)
best_wd, wdTrainingLosses, wdTrainingAcc, wdValidationLosses, wdValidationAcc = testHyperparameters("wd", weight_decay_rates)
with open("best_results.txt", "a") as f:
    f.write(f"\nBest Weight Decay: {best_wd}\n")
    f.write("WD Training Losses:\n")
    f.write(str(wdTrainingLosses) + "\n")
    f.write("WD Training Accuracies:\n")
    f.write(str(wdTrainingAcc) + "\n")
    f.write("WD Validation Losses:\n")
    f.write(str(wdValidationLosses) + "\n")
    f.write("WD Validation Accuracies:\n")
    f.write(str(wdValidationAcc) + "\n")

# Save weight decay results in a structured format (npz)
np.savez(
    "best_wd_results.npz",
    best_wd=best_wd,
    wdTrainingLosses=wdTrainingLosses,
    wdTrainingAcc=wdTrainingAcc,
    wdValidationLosses=wdValidationLosses,
    wdValidationAcc=wdValidationAcc,
    weight_decay_rates=weight_decay_rates
)

totalProgressTimer.timerBroadcast()
totalProgressTimer.endTimer()
# torch.save(model.state_dict(), "seismo_cnn_weights.pth")
# model = SeismoCNN(num_classes=4)  # Recreate the model architecture
# model.load_state_dict(torch.load("seismo_cnn_weights.pth"))
# model.to(device)  # Don't forget this if using CUDA
# model.eval()  # Set to evaluation mode for inference