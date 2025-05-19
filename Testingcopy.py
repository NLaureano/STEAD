import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

file_name = "merged/merge.hdf5"
csv_file = "merged/merge.csv"

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')
# filterering the dataframe
#df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 5) & (df.source_magnitude > 3)]
total = len(df)
print(f'total events selected: {total}')

# making a list of trace names for the selected data
traceName_list = df['trace_name'].to_list()
category_list = df['trace_category'].to_list()
magnitude_list = df['source_magnitude'].to_list()
#print(f"Length of matching earthquakes: {len(ev_list)}")
# retrieving selected waveforms from the hdf5 file: 
#dtfl = h5py.File(file_name, 'r')
noise = 0
noiseIndexes = list()
EqCount = 0
EqIndexes = list()
LowEqIndexes = list()
MedEqIndexes = list()
HighEqIndexes = list()

print(category_list[0])

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


for c, event in enumerate(category_list):
    traceName = traceName_list[c]
    if event == 'noise':
        noise += 1
        noiseIndexes.append((traceName, c, 'NOISE'))
    else:
        EqCount += 1
        thisMagnitude = magnitude_list[c]
        if thisMagnitude <= 2.0:
            LowEqIndexes.append((traceName, c, 'LOW'))
        elif thisMagnitude <= 4.0:
            MedEqIndexes.append((traceName, c, 'MED'))
        else:
            HighEqIndexes.append((traceName, c, 'HIGH'))
        EqIndexes.append(c)
    
saveList(noiseIndexes, 'noiseIndexes.txt')
saveList(LowEqIndexes, 'LowEqIndexes.txt')
saveList(MedEqIndexes, 'MedEqIndexes.txt')
saveList(HighEqIndexes, 'HighEqIndexes.txt')

# Define counts per split
train_counts = {'noise': 14000, 'low': 7000, 'mid': 7000, 'high': 7000}
val_counts   = {'noise': 2000,  'low': 1000, 'mid': 1000, 'high': 1000}
test_counts  = {'noise': 4000,  'low': 2000, 'mid': 2000, 'high': 2000}

# Slice each list accordingly
trainSet = (
    noiseIndexes[:train_counts['noise']] +
    LowEqIndexes[:train_counts['low']] +
    MedEqIndexes[:train_counts['mid']] +
    HighEqIndexes[:train_counts['high']]
)

validationSet = (
    noiseIndexes[train_counts['noise']:train_counts['noise'] + val_counts['noise']] +
    LowEqIndexes[train_counts['low']:train_counts['low'] + val_counts['low']] +
    MedEqIndexes[train_counts['mid']:train_counts['mid'] + val_counts['mid']] +
    HighEqIndexes[train_counts['high']:train_counts['high'] + val_counts['high']]
)

testSet = (
    noiseIndexes[-test_counts['noise']:] +
    LowEqIndexes[-test_counts['low']:] +
    MedEqIndexes[-test_counts['mid']:] +
    HighEqIndexes[-test_counts['high']:]
)

# Shuffle each set to mix categories
random.shuffle(trainSet)
random.shuffle(validationSet)
random.shuffle(testSet)

saveList(trainSet, 'trainSet.txt')
saveList(validationSet, 'validationSet.txt')
saveList(testSet, 'testSet.txt')


print(f"{noise} Noise / {EqCount} EQs ({len(LowEqIndexes)} Low, {len(MedEqIndexes)} Med, {len(HighEqIndexes)} High,")
# back_azimuth_deg 73.5
# coda_end_sample [[4713.]]
# network_code AV
# p_arrival_sample 900.0
# p_status manual
# p_travel_sec 8.329999923706055
# p_weight 0.5
# receiver_code AMKA
# receiver_elevation_m 116.0
# receiver_latitude 51.3771
# receiver_longitude 179.3
# receiver_type BH
# s_arrival_sample 1520.0
# s_status manual
# s_weight 0.5
# snr_db [58.09999847 61.40000153 63.        ]
# source_depth_km 58.14
# source_depth_uncertainty_km None
# source_distance_deg 0.02
# source_distance_km 1.78
# source_error_sec 1.2217
# source_gap_deg 139.496
# source_horizontal_uncertainty_km None
# source_id 13681341
# source_latitude 51.3734
# source_longitude 179.2752
# source_magnitude 3.5
# source_magnitude_author None
# source_magnitude_type mb
# source_mechanism_strike_dip_rake None
# source_origin_time 2009-06-14 08:40:01.85
# source_origin_uncertainty_sec 0.89
# trace_category earthquake_local
# trace_name AMKA.AV_20090614084000_EV
# trace_start_time 2009-06-14 08:40:01.180000