# Speech Emotion Recognition

This repository contains the code and resources for the Speech Emotion Recognition (SER) project, which aims to classify emotions from speech recordings using various machine learning and deep learning models.

## Project Overview

Speech Emotion Recognition is a challenging task in affective computing that involves identifying emotional states from vocal expressions. This project leverages advanced feature extraction techniques and machine learning methodologies to develop a robust SER model. The primary goal is to enhance human-computer interaction by enabling systems to recognize and respond to human emotions effectively.

## Datasets

The project utilizes two main datasets:

- **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS):** Comprises 7,356 recordings from 24 actors expressing various emotions.
- **Toronto Emotional Speech Set (TESS):** Contains 2,800 stimuli recorded by two actresses portraying seven different emotions.

These datasets provide a diverse range of emotional expressions necessary for training and evaluating emotion recognition models.

## Methods

The following models were explored in this project:

1. **K-Nearest Neighbors (KNN):** A baseline model that classifies data based on proximity in feature space.
2. **Multi-Layer Perceptron (MLP):** A neural network model that captures non-linear relationships in data.
3. **Convolutional Neural Network (CNN):** Utilizes convolutional layers to extract spatial features from audio spectrograms.
4. **Hybrid CNN-LSTM Model:** Combines CNN for spatial feature extraction with LSTM for capturing temporal dependencies.

## Experiment Setup

- **Feature Extraction:** Features such as Mel-frequency cepstral coefficients (MFCCs), chroma features, and mel spectrograms were extracted from audio recordings.
- **Data Augmentation:** Techniques like pitch shifting and time stretching were applied to improve model robustness.
- **Evaluation Metrics:** Models were evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

## Results

The hybrid CNN-LSTM model achieved the best performance with an accuracy of 68.4% and an F1-score of 72.4%, demonstrating its effectiveness in handling sequential datasets by leveraging both spatial and temporal features.

## Limitations and Future Work

- **Class Imbalance:** Addressing underrepresented emotions could improve generalization.
- **Dataset Bias:** Incorporating more diverse datasets can enhance applicability across cultures.
- **Real-world Testing:** Further testing in noisy environments is needed to ensure robustness.

Future work includes exploring multimodal emotion recognition by integrating visual cues and developing lightweight models for real-time applications.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python packages: `librosa`, `pandas`, `numpy`, `scikit-learn`, `keras`, etc.

### Installation

Clone this repository:

```bash
git clone https://github.ncsu.edu/svaidya6/engr-ALDA-Fall2024-P9.git
cd engr-ALDA-Fall2024-P9
```

### Install the Necessary Packages

To install the required Python packages for this project, run the following command in your terminal:

```bash
pip install -r requirements.txt
```
## Running the Code

### Preprocess the Data

- **Extract Audio Features:** Use the provided scripts to extract audio features necessary for training the models.

### Train the Models

- **Model Training:** Execute scripts to train various models including KNN, MLP, CNN, and Hybrid models.

### Evaluate Performance

- **Model Evaluation:** Utilize evaluation scripts to assess model accuracy and other performance metrics.

## Contributors

- Payal Mehta ([pmehta5@ncsu.edu](mailto:pmehta5@ncsu.edu))
- Shonil Bhide ([sbhide@ncsu.edu](mailto:sbhide@ncsu.edu))
- Shreya Vaidya ([svaidya6@ncsu.edu](mailto:svaidya6@ncsu.edu))