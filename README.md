# SeismoCNN – Seismic Event Classification with Deep Learning

This repository is a **modified fork of the official STEAD (STanford EArthquake Dataset) repository**. The original contents of the forked repository have been relocated to the `INFO/` directory. The current focus of this project is the **classification of seismic waveform data** into four categories: `Noise`, `Low`, `Medium`, and `High` magnitude earthquakes using a Convolutional Neural Network (CNN).
CNN development by Nicholas Laureano
---

## Prerequisites

To fully utilize this repository on a local machine, follow these steps:

1. **Follow the setup instructions from the original STEAD repository** (see `INFO/`).
2. **Download and include the full STEAD dataset**:
   - Ensure the `<100GB>` `merged/` directory containing the full labeled dataset is placed in the root directory of this repo.

---

## Execution Workflow

Once the `merged/` directory is in place, run the following scripts **in order**:

### 1. `GenerateTVTSets.py`
- Parses data from the `merged/` directory.
- Generates PyTorch datasets and index files for `Noise`, `Low`, `Medium`, and `High` waveform examples used in this study.
- **Output**: TXT files of dataset splits.

### 2. `CNNGenerator.py`
- Trains the CNN model using predefined hyperparameters.
- **Training time** (on NVIDIA GeForce RTX 3070 Ti): ~1.7 hours per model over 10 epochs.
- **Output**: Trained weights saved as `seismo_cnn_weights.pth`.

### 3. `CNNRun.py`
- Loads the trained model from `seismo_cnn_weights.pth`.
- Evaluates performance on the testing set.
- **Output**: Evaluation results printed to terminal and saved in `FinalEvaluationPrintout.txt`.

---

## Additional Files and Directories

| File/Directory         | Description |
|------------------------|-------------|
| `Tuning.py`            | Performs hyperparameter tuning on 8 different configurations to optimize model performance. (on NVIDIA GeForce RTX 3070 Ti): ~14.2 Hours|
| `plotting.py`          | Generates visualizations (accuracy/loss curves) of tuning results. Outputs are saved to `Pictures/`. |
| `TuningStats/`         | Stores `.npy` and `.txt` files with training/validation statistics from tuning runs. |
| `Pictures/`            | Contains saved plots of model performance across hyperparameter settings. |
| `GPUAvailable.py`      | Utility script to verify CUDA/GPU availability for PyTorch acceleration. |
| `TimeKeeper.py`        | Custom utility to estimate completion times during model training/tuning. |
| `SeismoCNNImport.py`   | Defines the CNN model architecture and handles STEAD dataset loading. |
| `viewingCenter.py`     | Utility script to visualize and verify the contents of the parsed datasets. |

---

## Citation

If using this work please cite the STEAD dataset:

Mousavi, S. M., Sheng, Y., Zhu, W., Beroza G.C., (2019).  STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI,  IEEE Access, doi:10.1109/ACCESS.2019.2947848
