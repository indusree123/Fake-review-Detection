# Enhancing Product Review Authenticity Detection with Ensemble Learning and BERT

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Introduction
This project addresses the growing challenge of identifying fake reviews in e-commerce platforms. We propose a novel framework integrating Term Frequency-Inverse Document Frequency (TF-IDF), Count Vectorizer, and Bidirectional Encoder Representations from Transformers (BERT) to enhance the detection of fraudulent reviews. By combining traditional feature extraction techniques with advanced deep learning models, our system improves both interpretability and scalability.

## Motivation
Fake reviews undermine consumer trust and skew product ratings. Current detection methods often fall short due to their limited contextual understanding. This project aims to strengthen review authenticity detection using ensemble learning models and BERT’s bidirectional context comprehension.

## Features
- Fake review detection using a hybrid approach.
- Integration of TF-IDF, Count Vectorizer, and BERT for feature extraction.
- Ensemble learning with classifiers such as SVM, Naïve Bayes, and Random Forest.
- Comprehensive evaluation using precision, recall, F1 score, and accuracy.

## Dataset
The dataset used consists of 2,501 product reviews from major UK grocery retailers and Amazon. It includes reviews categorized into verified and non-verified purchases. The dataset was sourced from Kaggle.

### Dataset Structure
- 32 attributes including product category, review rating, and verified purchase status.
- Target variable: Verified purchase (True/False).

## Methodology
1. **Data Preprocessing**: Involves tokenization, lowercasing, stop word removal, punctuation elimination, stemming, lemmatization, and normalization.
2. **Feature Extraction**:
   - TF-IDF and Count Vectorizer convert text to numerical features.
   - BERT for capturing contextual information.
3. **Model Training**: Ensemble models with SVM, Naïve Bayes, and Random Forest.
4. **Evaluation**: Performance metrics include accuracy, precision, recall, and F1-score.

## Technologies Used
- Python
- Scikit-learn
- TensorFlow/Keras
- BERT from the Huggingface Transformers library

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/enhancing-fake-review-detection.git
   cd enhancing-fake-review-detection
   ```
2. Install required libraries:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your dataset in CSV format.
2. Run the preprocessing script:
   ```sh
   python preprocess.py
   ```
3. Train the model:
   ```sh
   python train_model.py
   ```
4. Evaluate the results:
   ```sh
   python evaluate.py
   ```

## Results
| Feature Selection | Model Combination | Accuracy | Precision | Recall | F1-Score |
|-------------------|------------------|----------|-----------|--------|----------|
| TF-IDF            | SVM + RF         | 0.85     | 0.82      | 0.85   | 0.84     |
| Count Vectorizer  | SVM + NB         | 0.84     | 0.81      | 0.85   | 0.83     |
| BERT              | Single Model     | 0.84     | 0.85      | 0.84   | 0.85     |

## Conclusion
Our research demonstrates that combining TF-IDF, Count Vectorizer, and BERT with ensemble models provides a robust solution for detecting fake reviews. The proposed system improves accuracy and interpretability while offering scalability for real-world applications.

## Future Work
- Further fine-tuning of the BERT model.
- Incorporating additional behavioral features for enhanced accuracy.
- Exploring other transformer-based models.

## References
1. R. Mohawesh et al., "Fake reviews detection: A survey," *IEEE Access*, 2021.
2. H. Paul and A. Nikolaev, "Fake review detection on online platforms," *Data Mining and Knowledge Discovery*, 2021.
3. J. Mukherjee et al., "Fake review detection: Classification and analysis," *Technical Report*, 2013.

For a complete list of references, please refer to the [conference paper](https://ieeexplore.ieee.org/document/10808342).

