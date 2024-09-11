# Training ML Models to detect Diabetes from Retinopathy Images

## Table of Contents 🔍

1. 📜 Dataset Description
2. 🔧 ML Models Used
3. ✅ Workflow
4. 💻 Technologies Used
5. 📈 Results
6. 🔚 Conclusion
7. 📧 Contact

### 1. Dataset Description: 📜

The dataset consists of high-resolution retinal images designed for training and evaluating automated systems for Diabetic Retinopathy (DR) detection and grading. It includes images taken under various real-world clinical conditions, each assessed by a medical professional and graded on a scale of 0 to 1. Grade 0 indicates the presence of DR, while Grade 1 indicates its absence. This extensive dataset is valuable for developing and testing DR detection algorithms, offering diverse imaging conditions and expert-annotated grades that support robust training and performance evaluation.

It aims to enhance **early detection**, timely intervention, and personalized treatment for diabetes.

### 2. Machine Learning Models Used: 🔧

- ResNet50: A deep learning model with 50 layers, known for using residual connections to address vanishing gradients in deep networks.
- EfficientNet: A scalable model series that balances depth, width, and resolution to achieve high accuracy with fewer parameters.
- MobileNet V3: Designed for mobile and edge devices, this model is efficient and compact, using techniques like network architecture search and H-Swish activation.
- ViT (Vision Transformer): Applies transformer architecture to image recognition by processing image patches as sequences, capturing global information for improved performance.

### 3. Workflow: ✅

For each model, we:

- Used a data loader for training.
- Plotted the ROC curve and confusion matrix.
- Implemented early stopping at 100 epochs and plotted accuracy changes over epochs.storage

### 4. Technologies Used: 💻

A list of technologies that were used in the project are:

- Python (with Jupyter)
- ML Models: ResNet50, EfficientNet-b2, MobileNet V3, Vit
- More detailed tech/frameworks:

  1. **Matplotlib** (import matplotlib.pyplot as plt): A library for creating static, animated, and interactive visualizations in Python.

  2. **PIL (Python Imaging Library)** (from PIL import Image): A library for opening, manipulating, and saving image files.

  3. **Torch** (import torch, import torch.nn as nn, import torch.optim as optim): PyTorch library for tensor computations and building neural network models.

  4. **Torchvision** (import torchvision.transforms as transforms, from torchvision.datasets import ImageFolder, from torchvision import models): A library for computer vision tasks that provides datasets, model architectures, and image transformations.

  5. **Scikit-learn**(from sklearn.metrics import roc_curve, auc, confusion_matrix): A library for machine learning that includes tools for evaluating model performance with metrics such as ROC curve and confusion matrix.

  6. **Numpy** (import numpy as np): A library for numerical operations and handling arrays.

  7. **Seaborn** (import seaborn as sns): A library for statistical data visualization, built on top of Matplotlib.

### 5. Results: 📈📉

#### Model Accuracy Comparison:

- ResNet50: 95.67%
- EfficientNet-b2: 97.83%
- MobileNet V3: 97.83%
- ViT (Vision Transformer): 97.83%

All models performed well in diagnosing diabetic retinopathy (DR), with EfficientNet-b2, MobileNet V3, and ViT achieving the highest accuracy of 97.83%.

#### Sensitivity and Specificity Comparison:

- ROC-AUC Value: All models achieved 0.99, indicating near-perfect discriminatory power.
- Sensitivity: EfficientNet-b2, MobileNet V3, and ViT have similar sensitivity, while ResNet50 has slightly lower sensitivity.
- Specificity: ViT has the highest specificity, followed by EfficientNet-b2 and MobileNet V3, with ResNet50 slightly lower.

### 6. Conclusion 🔚

EfficientNet-b2, MobileNet V3, and ViT show strong and similar performance in sensitivity and specificity. ResNet50 has slightly lower performance in both metrics. ViT offers a balanced and strong performance overall.

**For more detailed results, please check the pdfs and codebase**

### 7. Contact 📧

If you are interested in this ML project or want to collaborate on other ML/ Visualization Projects (in Python), please do not hesitate to contact me!

**Fuad Jeet** - [jeet0001@umn.edu](mailto:jeet0001@umn.edu)

GitHub: [https://github.com/fuadmuhtasim](https://github.com/fuadmuhtasim)
