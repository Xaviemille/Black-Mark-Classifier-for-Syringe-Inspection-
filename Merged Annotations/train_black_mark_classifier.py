"""
train_black_mark_classifier.py

Binary classifier: black_mark vs hard_negative.
Uses ResNet-18 pretrained on ImageNet, fine-tuned on your syringe dataset.

Data sources:
  - merged_annotations.json: bounding box annotations for 9 labelled syringes
  - IMAGES_ROOT: all 43 run folders scanned for images

Labels:
  1 = black_mark    (image has at least one bounding box annotation)
  0 = hard_negative (image has no annotation — from known defective or unlabelled syringe)

Split strategy: stratified by syringe (run_folder), not by image,
to prevent data leakage between train/val/test sets.

"""

import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG 
ANNOTATIONS_PATH = r"C:\Users\HP\Downloads\Exported JSONs\merged_annotations.json"

# Root folder containing ALL 43 run subfolders
IMAGES_ROOT = r"C:\Users\HP\Downloads\Syringe Data Collection\syringe_images\B AND W MARKS (43 syringes)"

# Train/val/test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BATCH_SIZE  = 16
NUM_EPOCHS  = 20
LR          = 1e-3       # learning rate for head training phase
LR_FINETUNE = 1e-4       # learning rate for full fine-tune phase
HEAD_EPOCHS = 5          # epochs to train head only before unfreezing backbone

SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where to save the best model weights
MODEL_SAVE_PATH = r"C:\Users\HP\Downloads\black_mark_resnet18_best.pth"


random.seed(SEED)
torch.manual_seed(SEED)


# DATASET 

class SyringeDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of dicts with keys:
            'full_path'  (str): absolute path to image
            'run_folder' (str): syringe identity
            'label'      (int): 1 = black_mark, 0 = hard_negative
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["full_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, label