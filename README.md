# Spam Detection  

Spam Detection is a machine learning project that classifies SMS messages as either **spam** or **ham** (not spam). The project demonstrates the use of text preprocessing and classification algorithms to tackle the problem of spam filtering in real-world scenarios.  

## Features  
- Preprocessing and feature engineering for SMS message classification.  
- Utilizes machine learning algorithms for text classification tasks.  
- Provides performance metrics for model evaluation.  
- Interactive user interface using Streamlit for live predictions.  

## Dataset  
The dataset used for this project is `spam.csv`, containing labeled SMS messages:  
- **spam**: Messages identified as unwanted (promotional, malicious, etc.).  
- **ham**: Regular, non-spam messages.  

Features include:  
- `Label`: Classification (`spam` or `ham`).  
- `Message`: The SMS content.  

## Prerequisites  

Before running the Spam Detection project, ensure you have the following installed:  

1. **Python** (>=3.8)  
2. **Streamlit** (`pip install streamlit`)  
3. **Pandas** (`pip install pandas`)  
4. **NumPy** (`pip install numpy`)  
5. **Scikit-Learn** (`pip install scikit-learn`)  
6. **NLTK** (`pip install nltk`)  
7. **Matplotlib** (`pip install matplotlib`)  
8. **Seaborn** (`pip install seaborn`)  

## Installation  

1. **Clone the Repository**  
    ```bash  
    git clone https://github.com/abhimanyu726/spam_detection.git  
    cd spam_detection  
    ```  

2. **Install the Dependencies**  
    Install the dependencies using pip:  
    ```bash  
    pip install -r requirements.txt  
    ```  

## Usage  

1. **Run the Jupyter Notebook**  
   Open and run the Jupyter notebook `spam_detection.ipynb` for data exploration, model training, and evaluation:  
    ```bash  
    jupyter notebook spam_detection.ipynb  
    ```  

2. **Streamlit Interface**  
   For an interactive web-based prediction tool, use the Streamlit app:  
    ```bash  
    streamlit run app.py  
    ```  
   Enter the SMS message in the input field.  
   The model will classify the message as **spam** or **ham** based on the input.  

## Model  
The project employs several machine learning algorithms to achieve optimal predictions. Models used include:  
- Naive Bayes  
- Logistic Regression  
- Support Vector Machines (SVM)  

Each model’s performance is evaluated based on metrics like accuracy, precision, recall, and F1-score.  

## Results  
- **Best Model**: Naive Bayes achieved the highest performance in classifying SMS messages.  
- **Key Insights**: Preprocessing steps like removing stopwords and applying TF-IDF significantly improved model accuracy.  
