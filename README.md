# Tuberculosis WSI Classifier with TransMILx

This project implements an end-to-end pipeline for classifying tuberculosis in Whole Slide Images (WSIs) using:
- EfficientNet-based patch-level feature extraction
- Transformer-based MIL (Multiple Instance Learning) with TransMILx
- 5-fold cross-validation and test-time inference support

**Table of Contents:**
- [Tuberculosis WSI Classifier with TransMILx](#tuberculosis-wsi-classifier-with-transmilx)
  - [🧠 Core Features](#-core-features)
  - [📁 Project Structure](#-project-structure)
  - [🔧 Requirements](#-requirements)
  - [🧪 Inference on WSIs](#-inference-on-wsis)
  - [🏋️ Train TransMILx with 5-Fold Cross-Validation](#️-train-transmilx-with-5-fold-cross-validation)
  - [📈 Outputs](#-outputs)
  - [📬 Contact](#-contact)

---

## 🧠 Core Features

- Extract top-50 or above informative patches from WSIs.
- Use Vision Transformer (ViT) + Transformer Encoder for MIL classification.
- Evaluate model performance using accuracy, precision, recall, F1 score, Kappa, and AUC.


## 📁 Project Structure

```
├── turberclosis_patch.py         # WSI feature extractor + MIL inference
├── turberclosis_decision.py      # Model training with 5-fold cross-validation
├── data/                         # Folder for training data
├── model/                        # Folder for storing trained models
├── output/                       # Prediction and result outputs (patches, .pt features, CSVs)
├── results/                      # Evaluation metrics and plots
```


## 🔧 Requirements

- Python 3.8+
- PyTorch
- torchvision
- timm
- albumentations
- scikit-learn
- tqdm
- OpenSlide (for .tif WSI reading)

Install dependencies:
```bash
pip install -r requirements.txt
```


## 🧪 Inference on WSIs

To classify WSIs:

```bash
python turberclosis_patch.py
```

Configure at `turberclosis_patch.py`:
- `root_dir`: path to folders like `./data/positive/` and `./data/negative/` with `.tif` slides
- `model_path`: path to EfficientNet model
- `model2_path`: path to TransMILx model (`.pth`)


## 🏋️ Train TransMILx with 5-Fold Cross-Validation

```bash
python turberclosis_decision.py
```

This will train the model using cropped image patches and save the best model for each fold in `./results`.


## 📈 Outputs

- `output/classification_results.csv`: prediction summary for each slide
- `metrics_summary.csv`: performance metrics
- `results/`: contains confusion matrix and ROC curve for each fold


## 📬 Contact

For issues or questions, feel free to open an issue or reach out.

```
ljfirst@hanmail.net
```