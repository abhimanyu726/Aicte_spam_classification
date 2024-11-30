# Spam Detection Project
## Overview
This project is focused on building a machine learning model to classify SMS messages as spam or ham (not spam). Using the provided spam.csv dataset, we aim to analyze text data, extract meaningful features, and create a predictive model that can efficiently detect spam messages.

## Dataset
The dataset, spam.csv, contains SMS messages labeled as either:

-spam: Unwanted promotional or malicious messages.
-ham: Regular, non-spam messages.

## Dataset Details:
-Columns:
    -Label: Indicates whether the message is spam or ham.
    -Message: The content of the SMS message.
-File Size: Small-sized CSV file with labeled text data.

## Objectives
1.Data Exploration: Understand the distribution of spam and ham messages.
2.Preprocessing: Clean the data (remove punctuation, stopwords, etc.) and prepare it for model training.
3.Feature Engineering: Convert text to numerical features using techniques like:
    -TF-IDF
    -Bag of Words
    -Word Embeddings
4.Model Building: Train and evaluate classifiers such as:
    -Naive Bayes
    -Logistic Regression
    -Support Vector Machines (SVM)
5.Evaluation: Assess model performance using metrics like accuracy, precision, recall, and F1-score.

## Getting Started

### Prerequisites
1.**Python** 3.7 or higher
2.**Pandas**
3.**Numpy**
4.**Scikit-Learn**
5.**nltk** (for text preprocessing)

### Installation
Clone the repository and install the required dependencies:

    bash
    Copy code
    git clone <repository-url>
    cd <repository-folder>
    pip install -r requirements.txt

## Usage
Place the spam.csv file in the data/ folder.
Run the preprocessing script:
bash
Copy code
python preprocess.py
Train and evaluate the model:
bash
Copy code
python train_model.py
