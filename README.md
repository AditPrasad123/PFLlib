# Quantum-Enhanced Federated Skin Lesion Classification

This project compares a classical federated learning pipeline against a quantum-enhanced variant for skin lesion classification on the ISIC2019 dataset.

The goal is to understand whether quantum preprocessing and a quantum classifier head can improve performance, efficiency, or both, in a federated medical imaging setting.

## Project Highlights

- Federated learning with FedBABU for personalized client adaptation
- Quantum preprocessing via quanvolutional feature extraction
- Classical versus quantum comparison under the same medical classification task
- Analysis tools for ROC, PR curves, confusion matrices, Grad-CAM, t-SNE, and false-negative inspection

## Datasets

Two dataset variants are used in this project:

- `ISIC2019`: original image-based dataset used for the classical baseline
- `ISIC2019_quanv`: preprocessed quanvolutional dataset used for the quantum-enhanced pipeline

The quanv dataset is generated offline before training and is then used directly by the federated learning pipeline.

## Methods

### Quantum-Enhanced Pipeline

- Model: QuanvEfficientNetB0
- Federated algorithm: FedBABU
- Quantum component: QNN-based head / quantum feature processing
- Purpose: evaluate whether quantum feature extraction improves performance or efficiency

### Preprocessing Pipeline

The quantum preprocessing stage is applied before federated training and includes:

- Classical ISIC preprocessing on the source images
- Resize to `48 x 48`
- RGB-to-grayscale conversion
- Quanvolution with `2 x 2` patches and stride `2`
- Quantum circuit with `4` qubits
- RY encoding and entanglement via CNOT gates
- Output of a `4 x 24 x 24` quanv tensor per image

This means the FL model trains on precomputed quanv tensors rather than raw images.

## Experimental Setup

- Dataset: `ISIC2019` for the classical baseline
- Dataset: `ISIC2019_quanv` for the quantum-enhanced pipeline
- Number of clients: `6`
- Number of classes: `8`
- Federated algorithm: `FedBABU`
- Local epochs: `2`
- Batch size: `16`
- Local learning rate: `0.001`
- Device: `CUDA`
- FedBABU fine-tuning epochs: `10`
- Backbone: `EfficientNetB0`

## Results Summary

The results show a clear trade-off between predictive quality and runtime efficiency.

### Key Takeaway

The quantum pipeline is promising from an efficiency standpoint, but the classical model currently performs better on classification metrics. In its present form, quantum preprocessing does not yet surpass the classical baseline for this task, but it provides a strong direction for future hybrid quantum-classical work.

## Visual Results

This project includes the following types of analysis outputs:

- Confusion matrices
- ROC curves
- Precision-Recall curves
- Convergence plots
- Grad-CAM visualizations
- t-SNE projections
- False-negative analysis

## How to Run

Clone the repository and move into the project folder first:

```cmd
git clone https://github.com/AditPrasad123/PFLlib.git
cd PFLlib
```

Command for the quantum-enhanced FedBABU federated run:

```cmd
cd system 
python main.py -data ISIC2019_quanv -m QuanvEfficientNetB0 -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda
```

Command for the quantum-enhanced FedAvg federated run:

```cmd
cd system 
python main.py -data ISIC2019_quanv -m QuanvEfficientNetB0 -algo FedAvg -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda
```

Command for the quantum-enhanced FedProx federated run:

```cmd
cd system 
python main.py -data ISIC2019_quanv -m QuanvEfficientNetB0 -algo FedProx -gr 20 -ls 2 -lbs 16 -lr 0.001 -mu 0.01 -dev cuda
```

Command for the quantum-enhanced FedBABU federated run with ViT:

```cmd
cd system 
python main.py -data ISIC2019_quanv -m QuanvTinyViT -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda
```

Command for the classical FedBABU federated run:

```cmd
cd system 
python main.py -data ISIC2019 -m EfficientNetB0 -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda
```

Command for the FedBABU federated run with ViT model:

```cmd
cd system 
python main.py -data ISIC2019 -m TinyViT -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda
```

Example command for the centralized quantum-enhanced run:

```cmd
cd system/centralized
python quanv_efficientnetb0.py
```

Example command for the centralized quantum-enhanced run with ViT:

```cmd
cd system/centralized
python tinyvit.py
```

Example command for the centralized classical run:

```cmd
cd system/centralized
python efficientnetb0.py
```

Example command for the centralized classical run with ViT:

```cmd
cd system/centralized
python tinyvit.py
```

## Grad-CAM Usage

The Grad-CAM scripts support original-image overlays and filtering to only correctly classified samples with confidence above 0.7.

### Federated Grad-CAM

Run this from the `system` folder:

```cmd
python analysis/gradcam_fedbabu_personalized_clients.py --dataset ISIC2019_quanv --run_id 4 --overlay_on_original --only_correct --min_confidence 0.7
```

This generates Grad-CAM outputs for the personalized FedBABU client checkpoints and maps the heatmaps back to the original RGB image space when available.

### Centralized Grad-CAM

Run this from the `system` folder:

```cmd
python centralized/gradcam_quanv_efficientnetb0.py --dataset ISIC2019_quanv --only_correct --min_confidence 0.7 --overlay_on_original
```

This generates Grad-CAM outputs for the centralized QuanvEfficientNetB0 model and overlays the mapped heatmaps on the original image.

### Centralized Classical Grad-CAM

Run this from the `system` folder:

```cmd
python centralized/gradcam_efficientnetb0.py --dataset ISIC2019 --only_correct --min_confidence 0.7 --overlay_on_original
```

This generates Grad-CAM outputs for the centralized EfficientNetB0 model on ISIC2019 and overlays the heatmaps on the denormalized RGB input.

### Viewing Results for Federated Models

After model training is completed. Execute the following command.
```cmd
cd system
```

In line 16 of results.py file change the value of 'file_name' variable to store the name of the desired trained model as present in PFLlib/results folder. Then execute the following command.

```cmd
python results.py
```

### Viewing Results for Centralized Models

The results are automatically saved in a dedicated model-wise outputs folder under the PFLlib/system/centralized directory.

## Project Structure

- `system/`: training, analysis, and model code
- `dataset/`: ISIC2019 and ISIC2019_quanv data utilities
- `results/`: saved training curves, metrics, and visualizations
- `system/preprocessing_quanvolutionalCNN.py`: offline quanvolution preprocessing script

## Conclusion

This project presents a side-by-side comparison of a classical federated learning approach and a quantum-enhanced federated learning approach for skin lesion classification. The classical model currently achieves better predictive quality, while the quantum variant offers a runtime advantage. Together, these results motivate further exploration of quantum preprocessing and hybrid quantum-classical architectures in federated medical imaging.
