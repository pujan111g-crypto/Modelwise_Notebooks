# Learning from Less: Few-Shot Image Classification Using Transfer Learning

**A comparative study of pretrained deep learning models and a CNN trained from scratch for pet-breed classification with limited training data.**

## 📌 Overview

Deep learning models typically require large amounts of labeled data to achieve reliable image-classification performance. However, collecting and labeling sufficient data can be challenging in real-world applications.

This project investigates how effectively image-classification models can learn when only a limited number of training images are available per class.

Using the **Oxford-IIIT Pet Dataset**, we compare three ImageNet-pretrained architectures—**ResNet18, EfficientNet-B0, and MobileNetV3-Large**—against a **CNN trained from scratch**. The experiments evaluate model performance across different training-data sizes using accuracy, macro F1-score, and experiment runtime.

The goal is to understand the trade-off between classification performance, data efficiency, and computational cost.

## 🎯 Objectives

- Investigate the effect of limited training data on image-classification performance.
- Compare transfer learning with training a CNN from random initialization.
- Evaluate models using different numbers of training images per class.
- Compare test accuracy, macro precision, macro recall, macro F1-score, and experiment runtime.
- Examine whether pretrained feature representations improve learning when labeled data is scarce.
- Identify practical trade-offs between predictive performance and computational efficiency.

## 🐱 Dataset

**Oxford-IIIT Pet Dataset**

The dataset contains images of cats and dogs belonging to different breeds. For this project, five breeds are selected to create a manageable multi-class classification task.

| Property | Description |
|---|---|
| Dataset | Oxford-IIIT Pet Dataset |
| Selected classes | Abyssinian, Bengal, Persian, Pug, Siamese |
| Number of classes | 5 |
| Total selected images | 997 |
| Training images | 697 |
| Validation images | 150 |
| Test images | 150 |
| Input image size | 224 × 224 pixels |

### Data preparation

- Selected images belonging to the five chosen breeds.
- Combined the dataset's official `trainval` and `test` image pools before creating custom splits.
- Created a stratified 70:15:15 training, validation, and test split.
- Applied ImageNet normalization to the input images.
- Kept the validation and test sets fixed while varying the amount of training data.

**Important:** The splits used in this project are custom stratified splits of the combined image pools, not the original official test split.

## 🧠 Models Compared

### 1. ResNet18 — Transfer Learning

ResNet18 uses residual connections to support effective feature learning. Its ImageNet-pretrained feature extractor is frozen, and a new classification layer is trained for the five selected breeds.

### 2. EfficientNet-B0 — Transfer Learning

EfficientNet-B0 uses a compound scaling approach to balance model depth, width, and input resolution. Its pretrained feature extractor is frozen while the classification head is adapted to the selected classes.

### 3. MobileNetV3-Large — Transfer Learning

MobileNetV3-Large is designed for computational efficiency and mobile-friendly applications. This experiment evaluates whether its pretrained features can provide strong classification performance with limited labeled data.

### 4. CNN Trained from Scratch

A custom convolutional neural network is trained from random initialization without ImageNet-pretrained weights. Unlike the frozen-feature transfer-learning models, its trainable parameters learn image representations directly from the selected training images.

This provides a baseline for examining the benefits of pretrained visual features.

## 🔬 Experimental Methodology

### Few-shot training settings

The transfer-learning experiments evaluate the following numbers of training images **per class**:

| Images per class | Total training images |
|---:|---:|
| 5 | 25 |
| 10 | 50 |
| 25 | 125 |
| 50 | 250 |
| 100 | 500 |
| Full training set | 697 |

The few-shot subsets are selected exclusively from the training split. The validation and test sets remain unchanged across experiments.

The full-training-set experiment provides a baseline for comparison with the smaller training subsets.

### Training configuration

| Parameter | Configuration |
|---|---|
| Input resolution | 224 × 224 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-entropy loss |
| Training duration | 10 epochs |
| Model selection | Lowest validation loss |
| Pretrained models | ImageNet weights |
| Transfer-learning strategy | Frozen feature extractor + trainable classifier |

The CNN trained from scratch uses randomly initialized weights and trains its trainable network parameters. Architecture-specific implementation details may differ between models.

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

- **Test Accuracy:** Proportion of test images classified correctly.
- **Macro Precision:** Average precision across the five classes, giving each class equal weight.
- **Macro Recall:** Average recall across the five classes, giving each class equal weight.
- **Macro F1-score:** Harmonic mean of precision and recall, averaged equally across classes.
- **Experiment Runtime:** Time recorded for the experiment, including the operations covered by the implementation's timer.

Macro-averaged metrics are useful for understanding performance across individual breeds rather than allowing the largest class to dominate the evaluation.

## 📈 Results and Comparison

The following tables summarize the few-shot classification results for all four models under different training settings.

| Model | Training setting | Test accuracy | Macro F1-score | Experiment time |
|---|---|---:|---:|---:|
| ResNet18 | 5 images/class | 78.67% | 0.7852 | 8.02 s |
| ResNet18 | 10 images/class | 88.00% | 0.8753 | 7.82 s |
| ResNet18 | 25 images/class | 98.00% | 0.9800 | 10.33 s |
| ResNet18 | 50 images/class | 98.00% | 0.9799 | 13.76 s |
| ResNet18 | 100 images/class | 96.67% | 0.9669 | 21.52 s |
| ResNet18 | Full training set | 97.33% | 0.9735 | 38.07 s |
| EfficientNet-B0 | 5 images/class | 96.00% | 0.9604 | 7.53 s |
| EfficientNet-B0 | 10 images/class | 97.33% | 0.9736 | 9.06 s |
| EfficientNet-B0 | 25 images/class | 96.67% | 0.9666 | 11.03 s |
| EfficientNet-B0 | 50 images/class | 98.67% | 0.9867 | 14.81 s |
| EfficientNet-B0 | 100 images/class | 98.67% | 0.9867 | 23.66 s |
| MobileNetV3-Large | 5 images/class | 97.33% | 0.9731 | 7.67 s |
| MobileNetV3-Large | 10 images/class | 99.33% | 0.9933 | 7.42 s |
| MobileNetV3-Large | 25 images/class | 96.67% | 0.9667 | 10.08 s |
| MobileNetV3-Large | 50 images/class | 97.33% | 0.9736 | 13.49 s |
| MobileNetV3-Large | 100 images/class | 98.67% | 0.9867 | 21.38 s |
| CNN from scratch | 5 images/class | 22.00% | 0.2162 | 7.38 s |
| CNN from scratch | 10 images/class | 24.00% | 0.1883 | 6.78 s |
| CNN from scratch | 25 images/class | 36.00% | 0.3351 | 9.26 s |
| CNN from scratch | 50 images/class | 39.33% | 0.3680 | 12.53 s |
| CNN from scratch | 100 images/class | 43.33% | 0.4355 | 20.19 s |

### Key Observations

- **MobileNetV3-Large** achieved the highest observed test accuracy of **99.33%** with 10 images per class.
- **EfficientNet-B0** achieved **98.67%** accuracy with both 50 and 100 images per class.
- **ResNet18** achieved **98.00%** accuracy with 25 and 50 images per class.
- **CNN from scratch** achieved a maximum accuracy of **43.33%** with 100 images per class, highlighting the challenge of learning effective features from limited data without pretrained weights.
- Increasing the number of training images generally reduced validation loss, although test accuracy did not increase monotonically for every model.

**Note:** Experiment times are measured in seconds and may depend on the hardware and runtime environment. The full-training-set ResNet18 result is retained from the existing project results; corresponding full-training-set results for the other models have not been provided here.


### Initial observation

In the current ResNet18 experiment, the 25-image-per-class setting achieved **98.00% test accuracy and a macro F1-score of 0.9800**. The smaller 5- and 10-image settings performed less strongly.

These results illustrate that a pretrained model can perform well with a relatively small labeled training set. However, each setting was evaluated using a single sampled subset, and the test set contains 150 images. Small differences between settings should therefore be interpreted cautiously. Repeated runs with different random seeds would provide a more reliable estimate of performance.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **PyTorch** — Model development, training, and evaluation
- **Torchvision** — Pretrained architectures, image transforms, and dataset utilities
- **NumPy** — Numerical operations
- **Pandas** — Results organization and tabular analysis
- **Matplotlib** — Performance visualization
- **Scikit-learn** — Classification metrics and evaluation utilities
- **Google Colab** — Experiment execution

## 📁 Repository Structure

```text
learning-from-less-few-shot-classification/
│
├── notebooks/
│   ├── resnet18.ipynb
│   ├── mobilenetv3_large.ipynb
│   ├── efficientnet_b0.ipynb
│   └── cnn_from_scratch.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

*The filenames and folders above are a suggested structure. Adjust them to match the files actually present in the repository.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/learning-from-less-few-shot-classification.git
cd learning-from-less-few-shot-classification
```

Replace `YOUR-USERNAME` with the GitHub username that owns the repository.

### 2. Install dependencies

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
```

### 3. Open the notebooks

You can run the notebooks using Jupyter Notebook, JupyterLab, or Google Colab.

For Google Colab:

1. Open the desired notebook from the repository.
2. Upload it to Colab or open it through GitHub.
3. Configure the dataset path as required by the notebook.
4. Run the cells in order.
5. Save the notebook with outputs if you want to preserve the experiment results.

### 4. Dataset and checkpoints

The dataset must be downloaded or made accessible separately according to the dataset-loading code in the notebooks. It is not embedded in the `.ipynb` files.

Trained model checkpoints (`.pth` files) are also separate artifacts and are not automatically included when a notebook is uploaded. Large datasets and model weights can be kept outside GitHub when appropriate.

## 📉 Visualizations

The experiments generate visualizations to compare model performance and understand the effect of training-data availability.

Planned visualizations include:

- Test accuracy versus training images per class.
- Macro F1-score versus training images per class.
- Training and validation loss curves.
- Confusion matrices for model evaluation.
- Experiment runtime comparison across architectures.

The generated figures and experiment results will be organized in the `results/` directory.

## 💡 Key Research Questions

1. How does classification performance change as the number of labeled images per class decreases?
2. Do ImageNet-pretrained models outperform a CNN trained from scratch when training data is limited?
3. Which architecture offers the most favorable balance between accuracy, F1-score, and runtime?
4. Does the best-performing architecture remain the same across different training-data sizes?
5. How consistent are the results across repeated runs with different few-shot samples?

## ⚠️ Limitations

- Only five pet breeds are considered.
- The evaluation uses a relatively small custom test set.
- Results can vary depending on the sampled few-shot training images.
- A single run per setting does not establish statistical significance.
- Frozen-feature transfer learning and a fully trainable CNN differ in both initialization and training strategy.
- Runtime comparisons depend on hardware, software configuration, and the operations included in the timer.

## 🔮 Future Improvements

- Repeat experiments with multiple random seeds and report mean and standard deviation.
- Add confidence intervals or other uncertainty estimates where appropriate.
- Investigate partial fine-tuning of pretrained layers.
- Explore data augmentation and alternative learning rates.
- Compare model size, parameter count, and inference latency.
- Evaluate on additional classes or a larger independent test set.
- Investigate prototypical networks or other specialized few-shot learning methods.

## 👥 Contributors

This project was collaboratively developed, with each team member responsible for training and evaluating specific deep learning architectures.

- **Goghara Pujan** — Training and evaluation of MobileNetV3-Large.
- **Pandav Dhruv** — Training and evaluation of EfficientNet-B0.
- **Parmar Aryan** — Training and evaluation of CNN from Scratch and ResNet18.

All contributors worked toward comparing model performance for few-shot pet-breed classification using the Oxford-IIIT Pet Dataset.

## 📄 License

This project is distributed under the license specified in the repository's `LICENSE` file. If no license has been added yet, choose an appropriate license before describing the project as open source.

## ⭐ Acknowledgements

We acknowledge the creators of the Oxford-IIIT Pet Dataset and the developers of PyTorch and Torchvision for making image-classification research and experimentation accessible.

---

**Project:** Learning from Less: Few-Shot Image Classification Using Transfer Learning

**Focus:** Transfer learning · Few-shot image classification · CNNs · Computer vision · Model evaluation

