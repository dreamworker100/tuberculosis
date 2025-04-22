import os
dll_path = r'C:\\Users\\ljfir\\.conda\\envs\\tor\\Lib\\site-packages\\openslide\\openslide-bin-4.0.0.2-windows-x64\\bin'
# dll_path = r'C:\\Users\\nfs\\anaconda3\\envs\\tor2\\Lib\\site-packages\\openslide\\openslide-bin-4.0.0.6-windows-x64\\bin'
os.add_dll_directory(dll_path)
from openslide import open_slide
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from openslide import open_slide
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from heapq import nlargest
import csv
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SlideProcessor:
    def __init__(self, slide_path, model_path, patch_size=(224, 224), save_positive=False, output_jpg_dir=None, output_pt_dir=None):

        model_name = 'efficientnet_b0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = timm.create_model(model_name=model_name, num_classes=5).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.gmp = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling

        self.slide_path = slide_path
        self.patch_size_x, self.patch_size_y = patch_size
        self.save_positive = save_positive
        self.output_dir = output_jpg_dir
        self.output_pt_dir = output_pt_dir
        self.slide = open_slide(slide_path)
        self.large_w, self.large_h = self.slide.dimensions
        self.num_x = int(self.large_w / self.patch_size_x)
        self.num_y = int(self.large_h / self.patch_size_y)

        if output_jpg_dir is not None:
            os.makedirs(output_jpg_dir, exist_ok=True)

    def process_slide(self):
        all_features = []
        patch_id = 0
        start_time = time.time()
        positive_patches = []

        total_patches = self.num_x * self.num_y

        with tqdm(total=total_patches, desc="Processing slide patches", unit="patch") as pbar:
            for j in range(self.num_y):
                for i in range(self.num_x):
                    x = i * self.patch_size_x
                    y = j * self.patch_size_y
                    patch_id += 1

                    # Process the patch and obtain its feature and probability
                    patch_data = self.process_patch(x, y, patch_id)
                    if patch_data is not None:
                        feature, probability, img_bgr, patch_name = patch_data
                        all_features.append(feature)

                        # Track patches with predicted class 1 and their probabilities
                        if probability is not None:
                            positive_patches.append((probability, img_bgr, patch_name))

                    pbar.update(1)  # Update progress bar

        # Check if all_features is empty
        if len(all_features) == 0:
            print("No features were extracted from the slide.")
            return None  # Return early if no features are found

        # Concatenate all features into a single tensor
        try:
            all_features = torch.cat(all_features, dim=0)
        except Exception as e:
            print(f"Error while concatenating features: {e}")
            return None

        # Save only the top 50 positive patches by probability
        top_patches = nlargest(50, positive_patches, key=lambda x: x[0])  # Sort by probability
        for _, img_bgr, patch_name in top_patches:
            cv2.imwrite(os.path.join(self.output_dir, patch_name), img_bgr)

        end_time = time.time()
        print(f"Processing slide completed in {end_time - start_time:.2f} seconds.")

        # Save the extracted features to a file
        try:
            torch.save(all_features, self.output_pt_dir)
            print(f"Features saved successfully to {self.output_pt_dir}")
        except Exception as e:
            print(f"Error saving features to {self.output_pt_dir}: {e}")

        return all_features

    # def process_slide_thread(self, batch_size=128, max_workers=16):
    #     all_features = []
    #     patch_id = 0
    #     start_time = time.time()
    #
    #     total_patches = self.num_x * self.num_y
    #     patch_futures = []
    #     positive_patches = []
    #     patch_batch = []
    #
    #     # Use ThreadPoolExecutor to process patches in parallel
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         with tqdm(total=total_patches, desc="Processing slide patches", unit="patch") as pbar:
    #             for j in range(self.num_y):
    #                 for i in range(self.num_x):
    #                     x = i * self.patch_size_x
    #                     y = j * self.patch_size_y
    #                     patch_id += 1
    #
    #                     # Collect patches into a batch
    #                     patch_batch.append((x, y, patch_id))
    #
    #                     # Once the batch size is reached, process the batch
    #                     if len(patch_batch) == batch_size:
    #                         # Submit each patch in the batch for processing
    #                         for x, y, patch_id in patch_batch:
    #                             future = executor.submit(self.process_patch, x, y, patch_id)
    #                             patch_futures.append(future)
    #                         patch_batch = []  # Clear batch
    #
    #                 # Process the remaining patches after the loop
    #                 if patch_batch:
    #                     for x, y, patch_id in patch_batch:
    #                         future = executor.submit(self.process_patch, x, y, patch_id)
    #                         patch_futures.append(future)
    #                     patch_batch = []  # Clear last batch
    #
    #             # Process completed futures as they finish
    #             for future in as_completed(patch_futures):
    #                 patch_data = future.result()  # Return both the feature and probability
    #                 if patch_data is not None:
    #                     feature, predicted_class, probability, img_bgr, patch_name, x, y = patch_data
    #                     if feature is not None:
    #                         all_features.append(feature)
    #
    #                     # Track patches with predicted class 1 and their probabilities
    #                     if predicted_class == 1:
    #                         positive_patches.append((probability, img_bgr, patch_name))
    #
    #                 pbar.update(1)  # Update progress bar
    #
    #     # Check if all_features is empty
    #     if len(all_features) == 0:
    #         print("No features were extracted from the slide.")
    #         return None  # Return early if no features are found
    #
    #     # Concatenate all features into a single tensor
    #     try:
    #         all_features = torch.cat(all_features, dim=0)
    #     except Exception as e:
    #         print(f"Error while concatenating features: {e}")
    #         return None
    #
    #     # Save only the top 50 positive patches by probability
    #     top_patches = nlargest(50, positive_patches, key=lambda x: x[0])  # Sort by probability
    #     for probability, img_bgr, patch_name in top_patches:
    #         try:
    #             cv2.imwrite(os.path.join(self.output_dir, patch_name), img_bgr)
    #             print(f"Saved patch: {patch_name} with probability: {probability}")
    #         except Exception as e:
    #             print(f"Error saving image {patch_name}: {e}")
    #
    #     end_time = time.time()
    #     print(f"Processing slide completed in {end_time - start_time:.2f} seconds.")
    #
    #     # Save the extracted features to a file
    #     try:
    #         torch.save(all_features, self.output_pt_dir)
    #         print(f"Features saved successfully to {self.output_pt_dir}")
    #     except Exception as e:
    #         print(f"Error saving features to {self.output_pt_dir}: {e}")
    #
    #     return all_features
    def process_slide_thread(self, batch_size=128, max_workers=16):
        all_features = []
        patch_id = 0
        start_time = time.time()

        total_patches = self.num_x * self.num_y
        patch_futures = []
        positive_patches = []
        class_0_patches = []
        patch_batch = []

        # Use ThreadPoolExecutor to process patches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=total_patches, desc="Processing slide patches", unit="patch") as pbar:
                for j in range(self.num_y):
                    for i in range(self.num_x):
                        x = i * self.patch_size_x
                        y = j * self.patch_size_y
                        patch_id += 1

                        # Collect patches into a batch
                        patch_batch.append((x, y, patch_id))

                        # Once the batch size is reached, process the batch
                        if len(patch_batch) == batch_size:
                            # Submit each patch in the batch for processing
                            for x, y, patch_id in patch_batch:
                                future = executor.submit(self.process_patch, x, y, patch_id)
                                patch_futures.append(future)
                            patch_batch = []  # Clear batch

                # Process the remaining patches after the loop
                if patch_batch:
                    for x, y, patch_id in patch_batch:
                        future = executor.submit(self.process_patch, x, y, patch_id)
                        patch_futures.append(future)
                    patch_batch = []  # Clear last batch

                # Process completed futures as they finish
                for future in as_completed(patch_futures):
                    patch_data = future.result()  # Return both the feature and probability
                    if patch_data is not None:
                        feature, predicted_class, probability, img_bgr, patch_name, x, y = patch_data
                        if feature is not None:
                            all_features.append(feature)

                        # Track patches with their predicted class and probability
                        if predicted_class == 1:
                            positive_patches.append((probability, img_bgr, patch_name))
                        elif predicted_class == 0:
                            class_0_patches.append((probability, img_bgr, patch_name))

                    pbar.update(1)  # Update progress bar

        # Check if all_features is empty
        if len(all_features) == 0:
            print("No features were extracted from the slide.")
            return None  # Return early if no features are found

        # Concatenate all features into a single tensor
        try:
            all_features = torch.cat(all_features, dim=0)
        except Exception as e:
            print(f"Error while concatenating features: {e}")
            return None

        # Save only the top 50 positive patches by probability
        if len(positive_patches) < 50:
            # Sort class 0 patches by the highest probability of being class 1
            class_0_patches_sorted = nlargest(50 - len(positive_patches), class_0_patches, key=lambda x: x[0])

            # Add class 0 patches to fill up to 50 patches
            top_patches = positive_patches + class_0_patches_sorted
        else:
            # Only select the top 50 positive patches
            top_patches = nlargest(50, positive_patches, key=lambda x: x[0])

        for probability, img_bgr, patch_name in top_patches:
            try:
                cv2.imwrite(os.path.join(self.output_dir, patch_name), img_bgr)
                # print(f"Saved patch: {patch_name} with probability: {probability}")
            except Exception as e:
                print(f"Error saving image {patch_name}: {e}")

        end_time = time.time()
        print(f"Processing slide completed in {end_time - start_time:.2f} seconds.")

        # Save the extracted features to a file
        try:
            torch.save(all_features, self.output_pt_dir)
            print(f"Features saved successfully to {self.output_pt_dir}")
        except Exception as e:
            print(f"Error saving features to {self.output_pt_dir}: {e}")

        return all_features

    def process_slide_thread_500(self, batch_size=128, max_workers=16):
        all_features = []  # To store all features
        top_features = []  # To store only the top 500 features
        patch_id = 0
        start_time = time.time()

        total_patches = self.num_x * self.num_y
        patch_futures = []
        positive_patches = []
        class_0_patches = []

        # Batch processing using numpy arrays for image loading efficiency
        patch_batch = []

        # Use ThreadPoolExecutor to process patches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=total_patches, desc="Processing slide patches", unit="patch") as pbar:
                for j in range(self.num_y):
                    for i in range(self.num_x):
                        x = i * self.patch_size_x
                        y = j * self.patch_size_y
                        patch_id += 1

                        # Collect patches into a batch
                        patch_batch.append((x, y, patch_id))

                        # Once the batch size is reached, process the batch
                        if len(patch_batch) == batch_size:
                            # Process the batch concurrently
                            future = executor.submit(self.process_patch_batch, patch_batch)
                            patch_futures.append(future)
                            patch_batch = []  # Clear batch

                # Process the remaining patches after the loop
                if patch_batch:
                    future = executor.submit(self.process_patch_batch, patch_batch)
                    patch_futures.append(future)
                    patch_batch = []  # Clear last batch

                # Process completed futures as they finish
                for future in as_completed(patch_futures):
                    patch_data_list = future.result()
                    if patch_data_list:
                        for patch_data in patch_data_list:
                            if patch_data is not None:
                                feature, predicted_class, probability, img_bgr, patch_name, x, y = patch_data
                                if feature is not None:
                                    # Store all features
                                    all_features.append(feature)
                                    # Track patches with their predicted class and probability
                                    if predicted_class == 1:
                                        positive_patches.append((probability, feature, img_bgr, patch_name))
                                    elif predicted_class == 0:
                                        class_0_patches.append((probability, feature, img_bgr, patch_name))

                            pbar.update(1)  # Update progress bar

        # Save only the top 50 patches as images (optimized)
        top_50_image = self.save_top_50_patches(positive_patches, class_0_patches)

        # Stack and save all features and top 500 features as .pt files
        self.save_features(all_features, positive_patches, class_0_patches)

        end_time = time.time()
        print(f"Processing slide completed in {end_time - start_time:.2f} seconds.")

        try:
            all_features = torch.cat(all_features, dim=0)
        except Exception as e:
            print(f"Error while concatenating features: {e}")
            return None



        return top_50_image  # Return the top features tensor if needed

    def process_patch_batch(self, patch_batch):
        results = []
        for x, y, patch_id in patch_batch:
            result = self.process_patch(x, y, patch_id)
            results.append(result)
        return results

    def save_top_50_patches(self, positive_patches, class_0_patches):
        # Save only the top 50 patches as images
        if len(positive_patches) < 50:
            class_0_patches_sorted = nlargest(50 - len(positive_patches), class_0_patches, key=lambda x: x[0])
            top_patches = positive_patches + class_0_patches_sorted
        else:
            top_patches = nlargest(50, positive_patches, key=lambda x: x[0])

        all_patches = []  # To store the tensor format of patches
        for probability, _, img_bgr, patch_name in top_patches:
            try:
                # Convert BGR image to RGB and then to tensor format
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # Convert to HSV
                img_hsv = img_hsv / 255.0  # Normalize to [0, 1]

                # Convert the image to a tensor (H, W, C) → (C, H, W)
                tensor_image = torch.from_numpy(img_hsv).permute(2, 0, 1).float()
                all_patches.append(tensor_image)  # Append the converted tensor

                # Save the image (if needed)
                cv2.imwrite(os.path.join(self.output_dir, patch_name), img_bgr)
                print(f"Saved patch: {patch_name} with probability: {probability}")
            except Exception as e:
                print(f"Error processing or saving image {patch_name}: {e}")

        try:
            # Stack all patches into a single tensor with shape (50, C, H, W)
            tensor_50_patches = torch.stack(all_patches, dim=0)  # Shape: (50, C, H, W)
            tensor_50_patches = tensor_50_patches.unsqueeze(
                0)  # Add a batch dimension for consistency, (1, 50, C, H, W)
        except Exception as e:
            print(f"Error while stacking top 50 patches: {e}")
            return None

        return tensor_50_patches  # Return the tensor containing the top 50 patches

    def save_features(self, all_features, positive_patches, class_0_patches):
        # Save all features
        try:
            all_features = torch.stack(all_features, dim=0)
        except Exception as e:
            print(f"Error while stacking all features: {e}")
            return

        # Save the selected top 500 features
        if len(positive_patches) < 500:
            class_0_patches_sorted = nlargest(500 - len(positive_patches), class_0_patches, key=lambda x: x[0])
            top_patches = positive_patches + class_0_patches_sorted
        else:
            top_patches = nlargest(500, positive_patches, key=lambda x: x[0])

        top_features = [patch[1] for patch in top_patches]

        try:
            top_features = torch.stack(top_features, dim=0)
            top_features_file = self.output_pt_dir.replace(".pt", "_500.pt")
            torch.save(top_features, top_features_file)
            print(f"Top 500 features saved successfully to {top_features_file}")
        except Exception as e:
            print(f"Error while stacking or saving top 500 features: {e}")
            return

        # Save all features to the original .pt file
        try:
            torch.save(all_features, self.output_pt_dir)
            print(f"All features saved successfully to {self.output_pt_dir}")
        except Exception as e:
            print(f"Error saving all features to {self.output_pt_dir}: {e}")

    def process_patch(self, x, y, patch_id):
        img = self.slide.read_region((x, y), 0, (self.patch_size_x, self.patch_size_y)).convert('RGB')
        img = np.asarray(img).astype(np.uint8).copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Compute channel medians and the difference for filtering
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        b_median = np.median(b)
        r_median = np.median(r)
        g_median = np.median(g)
        diff = np.abs(r_median - (b_median + g_median) / 2)

        # Filter the patch by variance and noise
        variance = self.calculate_patch_variance(img)
        significant_pixels_ratio = self.calculate_rgb_difference(img)

        # Process the patch regardless of its class
        if diff > 2 and significant_pixels_ratio > 0.1 and not self.is_noisy_patch(img_bgr):
            try:
                img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                img_hsv = img_hsv / 255.0  # Normalize
                img_tensor = torch.FloatTensor(img_hsv).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_probs, predicted_classes = torch.max(probabilities, dim=1)

                    features = self.model.forward_features(img_tensor)
                    # Apply GAP and GMP to the extracted features
                    gap_features = self.gap(features).squeeze(-1).squeeze(-1)  # [B, feature_dim]
                    gmp_features = self.gmp(features).squeeze(-1).squeeze(-1)  # [B, feature_dim]
                    # Concatenate GAP and GMP features
                    concatenated_features = torch.cat((gap_features, gmp_features), dim=1)  # [B, 2 * feature_dim]

                # Return the concatenated feature, class, probability, image, and patch name for further processing
                probability = predicted_probs.item()
                patch_name = f"patch_x{x}_y{y}.jpg"
                return concatenated_features, predicted_classes.item(), probability, img_bgr, patch_name, x, y

            except cv2.error as e:
                print(f"Failed to process patch at ({x}, {y}). Error: {e}")
                return None

        return None  # If patch doesn't meet the criteria

    def is_noisy_patch(self, img, white_threshold=250, black_threshold=5, noise_threshold=0.05):
        """
        Determines if the patch is noisy or mostly blank.

        Args:
            img (ndarray): The image patch.
            white_threshold (int): The threshold for detecting nearly white patches.
            black_threshold (int): The threshold for detecting black pixels.
            noise_threshold (float): The ratio of black/white pixels to consider a patch noisy.

        Returns:
            bool: True if the patch is noisy or blank, False otherwise.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check if patch is almost white
        mean_intensity = np.mean(gray)
        if mean_intensity > white_threshold:
            return True

        # Check if patch contains mostly black and white pixels
        black_pixels = np.sum(gray < black_threshold)
        white_pixels = np.sum(gray > white_threshold)
        total_pixels = gray.size
        black_white_ratio = (black_pixels + white_pixels) / total_pixels

        return black_white_ratio > noise_threshold

    def calculate_patch_variance(self, img):
        """
        Calculate the variance of each color channel in the image and return the mean variance.
        """
        variances = np.var(img, axis=(0, 1))
        return np.mean(variances)

    def calculate_rgb_difference(self, img):
        """
        Calculate the number of pixels where the difference between the RGB channels is significant.
        """
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # Calculate absolute differences
        rg_diff = np.abs(r - g)
        rb_diff = np.abs(r - b)
        gb_diff = np.abs(g - b)

        # Threshold for considering a pixel as having a significant difference
        threshold = 10

        # Boolean arrays where True means the difference is significant
        significant_diff = (rg_diff > threshold) & (rb_diff > threshold) & (gb_diff > threshold)

        # Calculate the percentage of such pixels
        significant_pixels_ratio = np.sum(significant_diff) / (img.shape[0] * img.shape[1])

        return significant_pixels_ratio

class ViT_MIL(nn.Module):
    def __init__(self, num_classes=2, feature_dim=768, num_patches=50):
        super(ViT_MIL, self).__init__()

        # Load the pretrained Vision Transformer (ViT) model
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove the original classification head

        # Attention mechanism for MIL
        # self.attention = nn.Sequential(
        #     nn.Linear(feature_dim, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 1)
        # )
        self.attention = nn.Linear(feature_dim, 1)

        # Final classification layer for the bag
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, patches):

        if patches.dim() == 6 and patches.size(1) == 1:
            patches = patches.squeeze(1)  # 크기가 1인 두 번째 차원을 제거
        # Expecting input patches to have shape: (batch_size, 50, 3, 224, 224)
        batch_size, num_patches, _, H, W = patches.size()
        # assert num_patches == 10, "Expected 50 patches for each input."

        # Flatten the batch and patch dimensions for ViT input
        patches = patches.view(batch_size * num_patches, 3, H, W)

        # Extract features using the Vision Transformer
        features = self.vit(patches)  # (batch_size * num_patches, feature_dim)

        # Reshape features back to (batch_size, num_patches, feature_dim)
        features = features.view(batch_size, num_patches, -1)

        # Apply attention mechanism to get attention scores
        attention_weights = self.attention(features)  # (batch_size, num_patches, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize attention scores

        # Weighted sum of features to get bag-level representation
        weighted_sum = torch.sum(attention_weights * features, dim=1)  # (batch_size, feature_dim)

        # Final classification for each bag
        logits = self.classifier(weighted_sum)  # (batch_size, num_classes)

        return logits, attention_weights
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

def classify_slides_in_folder(root_dir, model_path, model2_path, num_patches=150000, batch_size=256, target_patch_size=224):
    slide_folders = ['positive', 'negative']
    results = []

    for slide_folder in slide_folders:
        folder_path = os.path.join(root_dir, slide_folder)
        label = 0 if slide_folder == 'negative' else 1  # 0 for negative, 1 for positive

        for slide_file in os.listdir(folder_path):
            if slide_file.endswith('.tif'):
                slide_path = os.path.join(folder_path, slide_file)
                positive_output_dir = f'./output/{slide_folder}/{slide_file[:-4]}'
                pt_dir = f'./output/pt/{slide_folder}/{slide_file[:-4]}.pt'


                os.makedirs(os.path.dirname(positive_output_dir), exist_ok=True)
                os.makedirs(os.path.dirname(pt_dir), exist_ok=True)

                # Check if the .pt file already exists, skip if it does
                if os.path.exists(pt_dir):
                    print(f"{pt_dir} already exists. Skipping processing for {slide_file}.")
                    continue  # Skip this slide if .pt file already exists

                print(f"Processing slide: {slide_file}...")
                processor = SlideProcessor(slide_path, model_path, patch_size=(target_patch_size, target_patch_size),
                                           save_positive=True, output_jpg_dir=positive_output_dir, output_pt_dir=pt_dir)

                # Extract features from the slide
                features = processor.process_slide_thread_500()  # Assuming you have this method
                print(features.shape)
                # Check if features are None or empty (quality issue or failed extraction)
                if features is None or (isinstance(features, torch.Tensor) and features.size(0) == 0):
                    print(f"Slide {slide_file} has low-quality patches or failed to extract features, marking as class 2.")
                    predicted_class = 2  # Class 2 for low-quality slide
                    probability_class_1 = 1.0  # Assume highest probability for low-quality class
                else:
                    # Sampling or padding features to get exactly num_patches
                    # if features.size(0) < num_patches:
                    #     indices = torch.randint(0, features.size(0), (num_patches,))
                    #     features = features[indices]
                    # else:
                    #     indices = torch.randperm(features.size(0))[:num_patches]
                    #     features = features[indices]

                    print("Feature extraction completed.")

                    # Load model2 and classify the extracted features

                    # model2 = ViT_MIL().to(device)
                    model2 = TransMILx(n_classes=2).to(device)

                    model2.load_state_dict(torch.load(model2_path, map_location=device))
                    model2.eval()

                    # Reshape features to (1, num_patches, feature_dim) before feeding it into model2
                    # features = features.unsqueeze(0).to(device)
                    features = features.to(device)
                    print('fff', features.shape)
                    with torch.no_grad():
                        outputs, attention_weights = model2(features)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        probability_class_1 = probabilities[:, 1].item()  # Probability of class 1

                # Store the result
                results.append([slide_file, label, predicted_class, probability_class_1])  # Store probability of class 1
                print(f"Slide {slide_file} classified as: {predicted_class}")

    # Save results to CSV
    save_results_to_csv(results, './output/classification_results.csv')

    # Calculate and print metrics
    calculate_and_print_metrics(results)



def save_results_to_csv(results, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Slide File', 'True Label', 'Predicted Label', 'Probability of Positive'])
        writer.writerows(results)


def calculate_and_print_metrics(results):
    true_labels = [row[1] for row in results]
    predicted_labels = [row[2] for row in results]
    predicted_probs = [row[3] for row in results]

    # Calculate various metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    # ROC-AUC is valid only if both classes are present in the dataset.
    try:
        roc_auc = roc_auc_score(true_labels, predicted_probs)
    except ValueError:
        roc_auc = 'N/A'

    # Print the metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"ROC AUC: {roc_auc}")

    # Save the metrics to a CSV file
    save_metrics_to_csv(accuracy, precision, recall, f1, kappa, roc_auc, './metrics_summary.csv')


def save_metrics_to_csv(accuracy, precision, recall, f1, kappa, roc_auc, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f"{accuracy * 100:.2f}%"])
        writer.writerow(['Precision', f"{precision:.4f}"])
        writer.writerow(['Recall', f"{recall:.4f}"])
        writer.writerow(['F1 Score', f"{f1:.4f}"])
        writer.writerow(['Cohen\'s Kappa', f"{kappa:.4f}"])
        writer.writerow(['ROC AUC', roc_auc])


# Main execution
if __name__ == '__main__':
    root_dir = 'g:/tuberclosis_400x_split'                      #  data directory
    model_path = './model/efficient_hsv_model_state_dict.pt'   # positive patch extracter
    model2_path = './model/transmilx.pth'                       # mil  classifier
    num_patches = 150000
    batch_size = 256
    target_patch_size = 224

    classify_slides_in_folder(root_dir, model_path, model2_path, num_patches=num_patches, batch_size=batch_size,
                              target_patch_size=target_patch_size)
