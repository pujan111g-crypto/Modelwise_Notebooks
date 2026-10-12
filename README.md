# Modelwise_Notebooks
Learning from Less: Few-Shot Image Classification Using Transfer Learning
Machine Learning project that studies how image-classification accuracy changes when models are trained with only a few labeled images.

Executive Summary
Three ImageNet-pretrained models (ResNet18, MobileNetV3-Large, EfficientNet-B0) used as frozen feature extractors were compared against a CNN trained from scratch. All models classified five pet breeds from the Oxford-IIIT Pet dataset, trained on 5, 10, 25, 50 and 100 images per class and on the full 697-image training split.
Key finding: Pretrained models reached >96% test accuracy with as few as 5–25 labeled images per class, while the CNN from scratch stayed below 45% in every few-shot setting and reached only 62.67% on the full training set.
Best accuracy-to-speed trade-off: MobileNetV3-Large — 98.67% test accuracy in 31.22 seconds.
Each setting was run once, so results are indicative rather than statistically conclusive.
1. Objective and Scope
Research question: How does image-classification performance change when models are trained with limited labeled data?
Approach: Compare three ImageNet-pretrained transfer-learning models against a randomly initialized CNN trained from scratch, across five few-shot data quantities plus a full-data baseline.
Workflow: Dataset → Preprocessing → Training with different data quantities → Validation → Testing → Model comparison.
2. Dataset
Five breeds from the Oxford-IIIT Pet dataset:
Breed
Example
Abyssinian

Bengal

Persian

Pug

Siamese

Total images: 997
Images from the original trainval and test partitions were combined and re-divided with a custom stratified 70/15/15 split (team’s own split, not the official dataset test split).
Split
Images
Purpose
Training
697
Images the model learns from
Validation
150
Monitors loss during training
Custom test
150
Held-out images used only for final evaluation
Total
997

Few-shot subsets
Setting
Images per class
Total training images
5-shot
5
25
10-shot
10
50
25-shot
25
125
50-shot
50
250
100-shot
100
500
Full dataset
~139
697
3. Models
Model
Type
Trainable params (approx.)
Notes
ResNet18
Transfer learning
Classifier head only
Frozen ImageNet backbone + new FC layer
MobileNetV3-Large
Transfer learning
Classifier head only
Inverted residuals + SE blocks
EfficientNet-B0
Transfer learning
Classifier head only
Compound scaling
CNN from Scratch
From scratch
~26M
4 convolutional blocks, no pretraining
Common training configuration
Input size: 224 × 224
Batch size: 16
Optimizer: Adam (lr = 0.001)
Epochs: 10
Loss: Cross-entropy
Backbone: frozen (transfer models) / fully trainable (scratch CNN)
4. Results
4.1 Overall Comparison
Model
Best Few-Shot
Full Dataset
At 5 images/class
Experiment Time
ResNet18
98.00% (25 & 50)
97.33%
78.67%
38.07 s
MobileNetV3-Large
99.33% (10)
98.67%
97.33%
31.22 s
EfficientNet-B0
98.67% (50 & 100)
98.67%
96.00%
35.59 s
CNN from Scratch
43.33% (100)
62.67%
22.00%
40.93 s
4.2 Detailed Few-Shot Accuracies
ResNet18
Images / class
Test Accuracy
5
78.67%
10
88.00%
25
98.00%
50
98.00%
100
96.67%
Full
97.33%
Just 25 images per class reached 98.00% — on par with the full-dataset 97.33%.
MobileNetV3-Large
Images / class
Test Accuracy
5
97.33%
10
99.33%
25
96.67%
50
97.33%
100
98.67%
Full
98.67%
Reached 99.33% with only 10 images/class — best few-shot of all models.
EfficientNet-B0
Images / class
Test Accuracy
5
96.00%
10
97.33%
25
96.67%
50
98.67%
100
98.67%
Full
98.67%
Strong from the start: 96% at 5 images/class, peaking at 98.67%.
CNN from Scratch
Images / class
Test Accuracy
5
22.00%
10
24.00%
25
36.00%
50
39.33%
100
43.33%
Full
62.67%
Far behind pretrained models: only 62.67% even with the full dataset.
4.3 Full-Dataset Metrics (detailed)
Model
Test Acc
Macro Precision
Macro Recall
Macro F1
Test Loss
Time
ResNet18
97.33%
0.9744
0.9733
0.9735
0.0899
38.07 s
MobileNetV3-Large
98.67%
—
—
0.9867
0.0560
31.22 s
EfficientNet-B0
98.67%
—
—
0.9867
0.0448
35.59 s
CNN from Scratch
62.67%
0.6380
0.6267
0.6238
1.1157
40.93 s
5. Key Takeaways
Pretrained ResNet18 hit 98.00% test accuracy with only 25 images per class.
More data did not always mean higher accuracy — ResNet18 scored 96.67% at 100 images vs 98.00% at 25 and 50.
Pretrained models crush the CNN from scratch — MobileNet hit 99.33% at 10 images/class while the scratch CNN stayed under 45% in all few-shot settings.
Every score comes from a separate held-out test set the models never trained on.
Closing statement
ImageNet-pretrained models with a frozen backbone reach >96% accuracy with as few as 5–25 labeled images per class, while a CNN trained from scratch remains far behind even with the full training set.
6. Repository Structure
.
├── README.md
├── notebooks/
│   ├── resNet18.ipynb
│   ├── MobileNetV3.ipynb
│   ├── EfficientNEt-B0.ipynb
│   └── CNN_From_Scratch.ipynb
├── docs/
│   ├── Learning_from_Less.pdf          # Full project report
│   └── slides/                         # Presentation slides
└── requirements.txt
7. How to Reproduce
# 1. Clone
git clone https://github.com/your-username/learning-from-less.git
cd learning-from-less

# 2. Create environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Open any notebook
jupyter notebook notebooks/resNet18.ipynb
Requirements (main packages)
torch
torchvision
numpy
matplotlib
scikit-learn
Pillow
tqdm
8. Team
Name
Enrollment No.
Dhruv Pandav
240280152045
Ghoghara Pujan
240280152023
Parmar Aryan
240280152048
9. License
This project is for academic purposes. Dataset: Oxford-IIIT Pet Dataset.
Learning from Less · Few-Shot Image Classification · 2026
