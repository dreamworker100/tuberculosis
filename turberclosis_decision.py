import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import random
import timm
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from shapely.geometry import MultiPoint
from shapely.affinity import scale
from shapely.geometry import Point
from shapely.affinity import translate
from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from torchvision import transforms
import pandas as pd
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from nystrom_attention import NystromAttention
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models
from sklearn.model_selection import StratifiedKFold, train_test_split
import math

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_accuracy(outputs, labels, batch_size):
    _, preds = outputs.max(1)
    correct = preds.eq(labels).sum()
    return correct.item() / batch_size
# class PositionalEncoding(nn.Module):
#     def __init__(self, dim, max_len=10000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return x
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
class TransMILx(nn.Module):
    def __init__(self, n_classes=2, dim_features=768, n_heads=8, n_layers=4, dropout=0.1):
        super(TransMILx, self).__init__()

        # ViT-B/16 backbone
        self.vit = models.vit_b_16(pretrained=True)
        # Remove the classification head
        self.vit.heads = nn.Identity()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=dim_features,
            nhead=n_heads,
            dim_feedforward=dim_features * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(dim_features, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim_features, dim_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_features // 2, n_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, n_patches, channels, height, width)
        batch_size, n_patches = x.size(0), x.size(1)

        # Reshape for processing patches
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        # Extract features using ViT
        features = self.vit(x)  # Shape: (batch_size * n_patches, dim_features)

        # Reshape back to separate patches
        features = features.view(batch_size, n_patches, -1)

        # Add positional encoding
        features = self.pos_encoder(features)

        # Apply transformer encoder
        features = self.transformer_encoder(features)

        # Global average pooling over patches
        features = torch.mean(features, dim=1)

        # Classification
        output = self.classifier(features)

        return output, output
class GeneDataset_crop2_50(Dataset):
    def __init__(self, root_folder, num_patches=20, transform=None):
        self.image_folder = root_folder
        self.num_patches = num_patches
        self.transform = transform
        self.image_files = []
        self.labels = {'negative': 0, 'positive': 1}  # Assuming '0' for normal and '1' for positive cases

        for label_folder in ['negative', 'positive']:
            folder_path = os.path.join(self.image_folder, label_folder)
            for gene_id in os.listdir(folder_path):
                gene_folder = os.path.join(folder_path, gene_id)
                if os.path.isdir(gene_folder):
                    files = [os.path.join(gene_folder, f) for f in os.listdir(gene_folder)]
                    self.image_files.append((files, self.labels[label_folder]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_paths, label = self.image_files[idx]

        if len(img_paths) == 0:
            # No images available, generate random placeholder images
            extended_img_paths = ['random'] * self.num_patches
        elif len(img_paths) < self.num_patches:
            # Repeat and randomly sample the images if fewer than num_patches are available
            extended_img_paths = img_paths * (self.num_patches // len(img_paths)) + random.sample(img_paths, self.num_patches % len(img_paths))
        else:
            # Randomly select num_patches images from the available ones
            extended_img_paths = random.sample(img_paths, self.num_patches)

        all_patches = []
        for img_path in extended_img_paths:
            if img_path == 'random':
                # Generate random image if no images are available
                image = np.random.rand(224, 224, 3).astype(np.float32) * 255  # Randomly generated image
            else:
                image = Image.open(img_path)
                if image is None:
                    print(f"Image at path {img_path} could not be loaded. Generating a random image.")
                    image = np.random.rand(224, 224, 3).astype(np.float32) * 255
                else:
                    try:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                    except Exception as e:
                        print(f"Failed to process image at {img_path}. Error: {e}")
                        image = np.random.rand(224, 224, 3).astype(np.float32) * 255

            if self.transform:
                image = self.transform(image)

            # Manually converting the image if no transform is provided
            image = np.asarray(image)
            if len(image.shape) == 2:  # For grayscale images
                image = np.stack([image] * 3, axis=-1)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Convert to HSV
            image = image / 255.0  # Normalize to [0, 1]

            tensor_image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to tensor (C, H, W)
            all_patches.append(tensor_image)

        stacked_features = torch.stack(all_patches)  # Stack all patches
        # print(stacked_features.shape)
        return stacked_features, label
def test_testset(model, test_loader, criterion, state='valid', save_dir='results'):
    model.eval()
    total_loss = 0
    total_acc = 0
    y_pred = []
    y_test = []
    y_prob = []

    pbar = tqdm(test_loader)
    for k, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        # labels = labels[:,19].long()
        with torch.no_grad():
            outputs, _ = model(images)
            # outputs, _, _ = model(images)
            # loss = criterion(outputs, labels)
            # total_loss += loss.item()
            _, predicted = outputs.max(1)

            # Detach from GPU and convert to CPU for sklearn metrics calculation
            predicted = predicted.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            # Store probabilities for ROC curve
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs = probs[:, 1].detach().cpu().numpy()  # Assuming binary classification

            # Append current batch's results to lists
            y_pred.extend(predicted)  # Use extend to flatten array into list
            y_test.extend(labels)     # Use extend to flatten array into list
            y_prob.extend(probs)

            pbar.set_description(f"{state} [{k}/{len(test_loader)}]")
            pbar.set_postfix(test_loss=f"{total_loss/(k+1):.4f}")

    pbar.close()
    # print("Unique labels in y_test:", np.unique(y_test))
    # print("Class distribution in y_test:", np.bincount(y_test))

    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    y_prob = np.array(y_prob)


    # Calculate performance metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred, average='binary', zero_division=1)
    recall_score = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    kappa_score = metrics.cohen_kappa_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Save the print output to a file
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f'{state}_results.txt')
    with open(log_path, 'w') as f:
        f.write(f'{state} accuracy={accuracy} precision_score={precision_score} recall_score={recall_score} f1_score={f1_score} kappa_score={kappa_score} roc_auc={roc_auc}\n')

    print(f'{state} accuracy={accuracy} precision_score={precision_score} recall_score={recall_score} f1_score={f1_score} kappa_score={kappa_score} roc_auc={roc_auc}')
    # Plot and save ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_path = os.path.join(save_dir, f'{state}_roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    # Plot and save Confusion Matrix with numbers
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(set(y_test)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Add the numbers
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(save_dir, f'{state}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    return accuracy, precision_score, recall_score, f1_score, kappa_score, roc_auc
def train_network_k(model, train_loader, val_loader, fold, model_name, num_epochs=10, learning_rate=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Training started...')
    class_weights = torch.tensor([0.5, 0.5], device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = HAMILLoss(lambda_reg=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, verbose=True)

    best_f1 = 0
    save_model_file = None
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        pbar = tqdm(train_loader)

        for k, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.long()
            labels = labels.to(device)

            optimizer.zero_grad()
            out, _ = model(images)
            loss = criterion(out, labels)
            # logits, instance_attn, bag_attn= model(images)
            # loss = criterion(logits, labels, instance_attn, bag_attn)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_acc += get_accuracy(out, labels, images.size(0))

            pbar.set_description(f"EPOCH[{epoch}][{k}/{len(train_loader)}]")
            pbar.set_postfix(train_loss=f"{train_loss/(k+1):.4f}", train_acc=f"{train_acc/(k+1):.4f}")

        # Validation phase
        accuracy, precision, recall, f1_score, kappa, roc_auc = test_testset(model, val_loader, criterion, state='valid')

        # Save the best model if the F1 score improves
        if f1_score >= best_f1:
            best_f1 = f1_score
            save_model_file = model_name + f'_{fold}_fold_best.pth'
            os.makedirs('./results', exist_ok=True)
            save_model_file = os.path.join('./results', save_model_file)
            print('saving ', save_model_file)
            torch.save(model.state_dict(), save_model_file)

        scheduler.step(f1_score)
        torch.cuda.empty_cache()
        pbar.close()

    return save_model_file  # Return the path of the best model from this fold
def train_5fold():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_folder = './tuberclosis_400x_split_2_50image'
    # train_folder = 'G:/2024_new_data_50image'
    train_folder = 'G:/2024_new_data_50image_2'
    # test_folder=  'G:/2024_new_data_50image'
    image_size = 224

    data_transforms_train = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),

    ], p=1.)

    data_transforms_test = A.Compose([
        A.Resize(image_size, image_size),
    ], p=1.)

    # Define transforms
    transform_train = lambda x: data_transforms_train(image=np.array(x))['image']
    transform_test = lambda x: data_transforms_test(image=np.array(x))['image']



    # Create dataset for training
    full_dataset = GeneDataset_crop2_50(train_folder, num_patches=50, transform=transform_train)
    # test_full_dataset = GeneDataset_crop2_50(test_folder, num_patches=50, transform=transform_test)
    from collections import Counter
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]
    print("Overall class distribution:", Counter(labels))

    # Define KFold cross-validator
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    model_name = "MICNet5aa"
    best_f1_score = 0
    best_model_path = None
    metrics_list = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_valid_ids, test_ids) in enumerate(kfold.split(full_dataset, labels)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_valid_labels = [labels[i] for i in train_valid_ids]
        train_ids, valid_ids = train_test_split(
            train_valid_ids,
            test_size=0.2,
            random_state=42 + fold,
            stratify=train_valid_labels
        )

        # Create subsets for training, validation, and testing
        train_labels = [labels[i] for i in train_ids]
        valid_labels = [labels[i] for i in valid_ids]
        test_labels = [labels[i] for i in test_ids]

        print("Train class distribution:", {label: train_labels.count(label) for label in set(train_labels)})
        print("Valid class distribution:", {label: valid_labels.count(label) for label in set(valid_labels)})
        print("Test class distribution:", {label: test_labels.count(label) for label in set(test_labels)})

        train_dataset = Subset(GeneDataset_crop2_50(train_folder, num_patches=50, transform=transform_train), train_ids)
        valid_dataset = Subset(GeneDataset_crop2_50(train_folder, num_patches=50, transform=transform_test), valid_ids)
        test_dataset = Subset(GeneDataset_crop2_50(train_folder, num_patches=50, transform=transform_test), test_ids)
        # Create training and validation datasets with different transforms


        # train_subsampler = full_dataset
        # valid_subsampler = test_full_dataset

        # Create data loaders for the fold
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



        model = TransMILx(n_classes=2).to(device)

        # Train the model
        train = True
        if train:
            # Train the model and return the path of the best model during this fold
            fold_best_model_path = train_network_k(model, train_loader, valid_loader, fold, model_name, num_epochs=20,
                                                   learning_rate=0.00001)

        # Load the best model from this fold
        model.load_state_dict(torch.load(fold_best_model_path, map_location=device))

        # Test the model on the validation set
        print('K-FOLD CROSS VALIDATION RESULTS:')
        print(f'Testing fold {fold} on validation data...')

        print('--------------------------------')
        accuracy, precision, recall, f1, kappa, roc_auc = test_testset(model, test_loader, criterion=nn.CrossEntropyLoss(), state=f'fold_{fold}',
                                        save_dir='./results')

        metrics_list.append((accuracy, precision, recall, f1, kappa, roc_auc))
        print(
            f"Fold {fold} metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}, Kappa={kappa}, roc_auc={roc_auc}")

    # Calculate average metrics across folds
    metrics_array = np.array(metrics_list)
    avg_metrics = metrics_array.mean(axis=0)

    print('Average Metrics Across Folds:')
    print(
        f"Accuracy={avg_metrics[0]:.4f}, Precision={avg_metrics[1]:.4f}, Recall={avg_metrics[2]:.4f}, F1={avg_metrics[3]:.4f}, Kappa={avg_metrics[4]:.4f} , roc_auc={avg_metrics[5]:.4f}")
    # Convert the results to a DataFrame for saving
    columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'ROC_AUC']
    results_df = pd.DataFrame(metrics_list, columns=columns)

    # Add a row for the average metrics
    average_row = pd.DataFrame([avg_metrics], columns=columns)
    average_row.index = ['Average']
    results_df = pd.concat([results_df, average_row])

    # Save the results to a CSV file
    csv_file_path = './results/metrics_results.csv'
    results_df.to_csv(csv_file_path, index_label='Fold')

    print(f'Results saved to {csv_file_path}')

    # Test the best model on the test set
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print('Final test on the best model across all folds...')
        # test_loader = DataLoader(PatchFeatureDataset('E:/AFB_2/tuberclosis_400x_patch_fet/test', max_patches=6000),
        #                          batch_size=4, shuffle=False)
        test_testset(model, valid_loader, criterion=nn.CrossEntropyLoss(), state='test', save_dir='./results')

if __name__ == '__main__':


    train_5fold()
