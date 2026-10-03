# Pneumonia Detection from Chest X-Ray Images

A deep-learning project for **binary classification of chest X-ray images** into:

- `NORMAL`
- `PNEUMONIA`

The project uses **Transfer Learning with PyTorch** and experiments with **ResNet18** and **DenseNet121**, progressively improving the model using data augmentation, class weighting, learning-rate scheduling, and Mixup augmentation.

> **Important:** This project is an educational machine-learning experiment. The model is not a medical diagnostic system and should not be used to make clinical decisions.

---

# 📌 Project Overview

The goal is to train a convolutional neural network that receives a chest X-ray image and predicts whether it belongs to one of two classes:

```text
Chest X-Ray
     │
     ▼
Deep Learning Model
     │
     ├───────────────┐
     ▼               ▼
 NORMAL          PNEUMONIA
```

The project starts with a basic **ResNet18 transfer-learning model** and then explores more advanced techniques to improve generalization.

---

# 🧠 Main Technologies

The project uses:

- **Python**
- **PyTorch**
- **Torchvision**
- **DenseNet121**
- **ResNet18**
- **Transfer Learning**
- **CrossEntropyLoss**
- **Adam / AdamW**
- **Cosine Annealing Learning Rate Scheduler**
- **Data Augmentation**
- **Mixup**
- **Class Weighting**
- **Scikit-learn**
- **Google Colab**
- **CUDA / GPU acceleration**

---

# 📊 Dataset

The notebook uses the **Chest X-Ray Pneumonia** dataset downloaded through Kaggle.

The dataset contains two classes:

```text
NORMAL
PNEUMONIA
```

The dataset is downloaded using the Kaggle API:

```python
from google.colab import files
import os

# Upload kaggle.json
uploaded = files.upload()

# Configure Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract dataset
!unzip -q chest-xray-pneumonia.zip -d chest_xray_data

print("Dataset downloaded and unzipped successfully!")
```

The extracted dataset is expected under:

```text
chest_xray_data/
└── chest_xray/
    ├── train/
    ├── test/
    └── val/
```

---

# 🔄 Dataset Classes

The notebook confirms the following classes:

```text
['NORMAL', 'PNEUMONIA']
```

Because this is a **binary multi-class classification problem**, every image receives exactly one class label.

For example:

```text
Image A → NORMAL
Image B → PNEUMONIA
```

This differs from multi-label classification, where an image could belong to multiple classes simultaneously.

---

# 1. Data Preprocessing & Augmentation

The first implementation uses the following training transformations:

```python
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Training Augmentation

Training images receive:

- Resize to `224 × 224`
- Random horizontal flip
- Random rotation up to 10 degrees
- Tensor conversion
- ImageNet normalization

Validation/test images do not receive random augmentation.

---

# 2. Load the Dataset

```python
train_dir = 'chest_xray_data/chest_xray/train'
test_dir = 'chest_xray_data/chest_xray/test'

train_dataset = ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_dataset = ImageFolder(
    root=test_dir,
    transform=val_test_transform
)
```

The dataset is loaded using PyTorch's:

```python
ImageFolder
```

which automatically assigns class IDs based on the folder structure.

---

# 3. Create DataLoaders

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

The training set is shuffled while the test set is not.

---

# 🏗️ Experiment 1 — ResNet18 Baseline

The first model uses a pre-trained **ResNet18**.

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Load pre-trained ResNet18
weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)
```

---

## Replace the Classification Head

The original ResNet18 classification layer is designed for ImageNet.

We replace it with a two-class output layer:

```python
num_ftrs = model.fc.in_features

model.fc = nn.Linear(
    num_ftrs,
    2
)
```

The resulting architecture is:

```text
Pre-trained ResNet18
        │
        ▼
Feature Extractor
        │
        ▼
Fully Connected Layer
        │
        ▼
2 Outputs
   ┌────┴────┐
   ▼         ▼
NORMAL   PNEUMONIA
```

---

# ⚙️ Loss Function & Optimizer

Because this is a two-class multi-class classification problem, the notebook uses:

```python
criterion = nn.CrossEntropyLoss()
```

The optimizer is:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)
```

---

# 🏋️ Training

The baseline model is trained for:

```python
num_epochs = 10
```

During training, the notebook tracks:

- Training loss
- Training accuracy

```python
for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, preds = torch.max(
            outputs,
            1
        )

        correct += (
            (preds == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] - "
        f"Train Loss: {epoch_loss:.4f} - "
        f"Train Acc: {epoch_acc * 100:.2f}%"
    )
```

---

# 📈 ResNet18 Results

The notebook produced the following training results:

| Epoch | Training Loss | Training Accuracy |
|---:|---:|---:|
| 1 | 0.0254 | 99.14% |
| 2 | 0.0209 | 99.25% |
| 3 | 0.0198 | 99.35% |
| 4 | 0.0243 | 99.21% |
| 5 | 0.0175 | 99.46% |
| 6 | 0.0136 | 99.56% |
| 7 | 0.0211 | 99.33% |
| 8 | 0.0101 | 99.64% |
| 9 | 0.0165 | 99.37% |
| 10 | 0.0147 | 99.50% |

The final test accuracy reported by the notebook was:

```text
84.13%
```

This large difference between training and test accuracy indicates that the baseline model was not generalizing as well as its training accuracy suggested.

---

# 💾 Save the ResNet18 Model

The trained model can be saved using:

```python
torch.save(
    model.state_dict(),
    'pneumonia_resnet18.pth'
)

print("Model saved successfully!")
```

---

# 🔍 Inference on a New X-Ray

The notebook also supports uploading a new X-ray image in Google Colab.

```python
from google.colab import files
import io

print("Please upload a chest X-ray image:")

uploaded = files.upload()
```

The uploaded image is transformed using the same preprocessing used during testing:

```python
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

The model then produces class probabilities using Softmax:

```python
with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        1
    )
```

Example recorded prediction:

```text
Prediction: NORMAL
Confidence: 71.39%
```

---

# 🧪 Experiment 2 — Improving the Model

The notebook then introduces several techniques to improve performance.

The second approach uses:

- DenseNet121
- Stronger data augmentation
- Class weighting
- Dropout
- AdamW
- L2 regularization
- Cosine Annealing learning-rate scheduling
- Best-model checkpointing

---

# 🧠 DenseNet121

The model is changed from ResNet18 to a pre-trained DenseNet121:

```python
from torchvision.models import (
    densenet121,
    DenseNet121_Weights
)

weights = DenseNet121_Weights.DEFAULT

model = densenet121(
    weights=weights
)
```

The final classifier is replaced:

```python
num_features = model.classifier.in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 2)
)
```

The classifier therefore becomes:

```text
DenseNet121
     │
     ▼
Dropout(0.4)
     │
     ▼
Linear Layer
     │
     ▼
2 Classes
```

---

# 🎨 Enhanced Data Augmentation

The second experiment uses stronger augmentation:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

The additional transformations include:

- Rotation up to 15 degrees
- Brightness variation
- Contrast variation
- Small translations

---

# ⚖️ Handling Class Imbalance

The training dataset contains:

```text
NORMAL     = 1341
PNEUMONIA  = 3875
```

Therefore, the classes are not evenly distributed.

The notebook calculates inverse-frequency class weights:

```python
targets = [
    label
    for _, label in train_dataset.samples
]

class_counts = np.bincount(
    targets
)

class_weights = (
    1.0 /
    torch.tensor(
        class_counts,
        dtype=torch.float
    )
)

class_weights = (
    class_weights /
    class_weights.sum()
)

class_weights = class_weights.to(device)
```

The calculated weights were:

```text
NORMAL     → 0.7429
PNEUMONIA  → 0.2571
```

These weights are then passed to the loss function:

```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)
```

This gives more importance to the minority class during optimization.

---

# ⚙️ Optimizer & Scheduler

The second experiment uses **AdamW**:

```python
optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=3e-4,
    weight_decay=1e-3
)
```

A cosine learning-rate scheduler is also used:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)
```

The scheduler is updated after every epoch:

```python
scheduler.step()
```

---

# 📈 DenseNet121 Results

The second experiment achieved:

```text
Highest Test Accuracy: 91.19%
```

The training progression was:

| Epoch | Train Loss | Train Accuracy | Test Accuracy |
|---:|---:|---:|---:|
| 1 | 0.2209 | 90.68% | 87.50% |
| 2 | 0.1109 | 95.74% | 88.46% |
| 3 | 0.0867 | 96.91% | 88.94% |
| 4 | 0.0729 | 97.26% | 90.06% |
| 5 | 0.0791 | 97.18% | **91.19%** |
| 6 | 0.0695 | 97.58% | 87.66% |
| 7 | 0.0658 | 97.49% | 89.26% |
| 8 | 0.0640 | 97.68% | 89.10% |
| 9 | 0.0631 | 97.66% | 89.10% |
| 10 | 0.0582 | 97.68% | 89.74% |

The best-performing checkpoint is saved as:

```text
best_pneumonia_densenet.pth
```

---

# 🧪 Experiment 3 — DenseNet121 + Mixup

The third experiment introduces a more extensive training setup.

It uses:

- DenseNet121
- Stratified train/test splitting
- Mixup augmentation
- AdamW
- Weight decay
- Cosine Annealing
- Best-model checkpointing
- Precision
- Recall
- F1-score
- Confusion matrix

---

# 📚 Stratified Dataset Split

The notebook combines the existing train, test, and validation datasets:

```python
full_dataset = torch.utils.data.ConcatDataset([
    full_train,
    full_test,
    full_val
])
```

The combined dataset contains:

```text
Total images: 5856
```

It is then divided using an **80/20 stratified split**:

```python
train_idx, test_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.20,
    shuffle=True,
    stratify=targets,
    random_state=42
)
```

Result:

```text
Total images: 5856
Train split: 4684
Test split: 1172
```

Stratification preserves the relative class distribution across the two splits.

---

# 🔀 Mixup Augmentation

Mixup is used to create new training examples by combining two images and their labels.

The notebook implements:

```python
def mixup_data(x, y, alpha=0.2):

    if alpha > 0:
        lam = np.random.beta(
            alpha,
            alpha
        )
    else:
        lam = 1.0

    batch_size = x.size(0)

    index = torch.randperm(
        batch_size
    ).to(device)

    mixed_x = (
        lam * x
        +
        (1 - lam) * x[index]
    )

    y_a, y_b = y, y[index]

    return (
        mixed_x,
        y_a,
        y_b,
        lam
    )
```

The corresponding loss is calculated as:

```python
def mixup_criterion(
    criterion,
    pred,
    y_a,
    y_b,
    lam
):

    return (
        lam * criterion(pred, y_a)
        +
        (1 - lam) * criterion(pred, y_b)
    )
```

This encourages the model to learn smoother decision boundaries.

---

# 🏗️ DenseNet121 Architecture

The third experiment again uses DenseNet121:

```python
weights = DenseNet121_Weights.DEFAULT

model = densenet121(
    weights=weights
)

for param in list(
    model.parameters()
)[:-30]:

    param.requires_grad = False

num_features = (
    model.classifier.in_features
)

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(
        num_features,
        2
    )
)

model = model.to(device)
```

Only the later portion of the network is allowed to update while earlier layers remain frozen.

---

# 🏋️ Training Configuration

The third experiment uses:

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=3e-4,
    weight_decay=1e-3
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)
```

Training is performed for:

```python
num_epochs = 10
```

The best model is saved whenever the test accuracy improves:

```python
if test_acc > best_acc:

    best_acc = test_acc

    torch.save(
        model.state_dict(),
        'best_pneumonia_mixup.pth'
    )
```

---

# 📈 Experiment 3 Results

The third experiment produced the following results:

| Epoch | Train Loss | Train Accuracy | Test Accuracy |
|---:|---:|---:|---:|
| 1 | 0.2920 | 87.99% | 95.14% |
| 2 | 0.2199 | 92.11% | 96.50% |
| 3 | 0.1877 | 93.72% | **96.67%** |
| 4 | 0.1868 | 94.01% | 96.59% |
| 5 | 0.1640 | 95.30% | 96.50% |
| 6 | 0.1646 | 94.75% | 96.42% |
| 7 | 0.1712 | 94.19% | 96.59% |
| 8 | 0.1666 | 94.42% | 96.59% |
| 9 | 0.1569 | 94.86% | **96.67%** |
| 10 | 0.1467 | 95.28% | 96.50% |

The highest test accuracy reported in this experiment was:

```text
96.67%
```

---

# 📊 Final Classification Metrics

After training, the best checkpoint is loaded:

```python
model.load_state_dict(
    torch.load(
        'best_pneumonia_mixup.pth'
    )
)
```

The notebook then generates a classification report.

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.95 | 0.92 | 0.94 | 317 |
| PNEUMONIA | 0.97 | 0.98 | 0.98 | 855 |
| **Overall Accuracy** | | | **0.97** | **1172** |

The reported averages were:

```text
Macro Average:
Precision = 0.96
Recall    = 0.95
F1-score  = 0.96

Weighted Average:
Precision = 0.97
Recall    = 0.97
F1-score  = 0.97
```

---

# 🧮 Confusion Matrix

The final confusion matrix was:

```text
TN: 293
FP: 24

FN: 15
TP: 840
```

Represented as a matrix:

```text
                 Predicted
              NORMAL  PNEUMONIA
Actual NORMAL    293       24
Actual PNEUMONIA  15      840
```

Where:

- **TN (True Negative)** = 293
- **FP (False Positive)** = 24
- **FN (False Negative)** = 15
- **TP (True Positive)** = 840

---

# 🔍 Final Model Inference

The final model can also classify an uploaded chest X-ray.

The same preprocessing is applied:

```python
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

The model generates probabilities using:

```python
probabilities = torch.softmax(
    outputs,
    dim=1
)
```

The class with the highest probability is selected:

```python
confidence, predicted_class = torch.max(
    probabilities,
    1
)
```

The notebook uses:

```python
class_names = [
    'NORMAL',
    'PNEUMONIA'
]
```

---

# 🧪 Example Prediction

One recorded inference result was:

```text
Uploaded File: images-normal (1).jpg
Prediction: NORMAL
Confidence: 71.39%
```

Another recorded inference result was:

```text
Uploaded File: ijap5072666-fig-0002b-m (2).jpg
Prediction: PNEUMONIA
Confidence: 92.50%
```

These examples demonstrate the model's inference pipeline, but individual predictions should not be interpreted as medical diagnoses.

---

# 📦 Model Checkpoints

The notebook saves several model checkpoints.

## ResNet18

```text
pneumonia_resnet18.pth
```

## DenseNet121 — Improved Model

```text
best_pneumonia_densenet.pth
```

## DenseNet121 — Mixup Model

```text
best_pneumonia_mixup.pth
```

The final Mixup experiment achieved the highest reported test accuracy in the notebook.

---

# 📊 Experiment Comparison

The progression of the experiments can be summarized as:

| Experiment | Model | Main Techniques | Best Test Accuracy |
|---|---|---|---:|
| 1 | ResNet18 | Basic Transfer Learning | 84.13% |
| 2 | DenseNet121 | Augmentation + Class Weights + Dropout + AdamW + Scheduler | 91.19% |
| 3 | DenseNet121 | Mixup + Stratified Split + AdamW + Scheduler | **96.67%** |

The experiments show how changing the architecture and training strategy affected the reported test performance.

---

# 🔬 Techniques Explored

This project demonstrates several deep-learning techniques.

## Transfer Learning

Pre-trained ImageNet models are reused as feature extractors.

Models explored:

```text
ResNet18
DenseNet121
```

---

## Data Augmentation

The project uses:

```text
RandomHorizontalFlip
RandomRotation
ColorJitter
RandomAffine
```

to introduce variation into the training images.

---

## Class Weighting

Class weights are calculated from the training distribution to give additional importance to the minority class.

---

## Dropout

The DenseNet121 classifier uses:

```python
nn.Dropout(0.4)
```

to reduce over-reliance on individual neurons.

---

## AdamW

AdamW is used with:

```text
Learning Rate = 3e-4
Weight Decay = 1e-3
```

---

## Cosine Annealing

The learning rate is adjusted using:

```python
CosineAnnealingLR
```

over the 10 training epochs.

---

## Mixup

Mixup creates synthetic training examples by combining images and labels.

This was used in the third experiment.

---

## Stratified Splitting

The third experiment uses:

```text
80% Training
20% Testing
```

with stratification to preserve class proportions.

---

# 🧠 Complete Pipeline

The final experimental pipeline can be represented as:

```text
                 Chest X-Ray Dataset
                         │
                         ▼
                  Image Preprocessing
                         │
                         ▼
                 Stratified 80/20 Split
                         │
                         ▼
                    Mixup Augmentation
                         │
                         ▼
                  Pre-trained DenseNet121
                         │
                         ▼
                    Frozen Early Layers
                         │
                         ▼
                     Dropout 0.4
                         │
                         ▼
                   2-Class Classifier
                         │
                         ▼
                  CrossEntropyLoss
                         │
                         ▼
                       AdamW
                         │
                         ▼
                 Cosine Annealing LR
                         │
                         ▼
                       Training
                         │
                         ▼
                 Best Model Checkpoint
                         │
                         ▼
                 Final Evaluation
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Classification Report    Confusion Matrix
             │
             ▼
       Model Inference
             │
             ▼
      NORMAL / PNEUMONIA
```

---

# 📚 Key Concepts Demonstrated

This project covers:

- Binary image classification
- Transfer learning
- CNNs
- ResNet18
- DenseNet121
- Image preprocessing
- Data augmentation
- Class imbalance
- Class weighting
- Cross-Entropy Loss
- Adam optimizer
- AdamW optimizer
- Weight decay
- Dropout
- Cosine Annealing
-
