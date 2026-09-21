# 📰 Fake News Detection using NLP

A machine learning project that classifies news articles as **Real** or **Potentially Misleading** using Natural Language Processing (NLP).

The project uses TF-IDF for text feature extraction and compares multiple machine learning models, including Logistic Regression, Naive Bayes, and Linear SVM.

## 🌐 Live Demo

Try the deployed Streamlit application:

https://fake-news-detection-nlp-bahdmekqtglhdmot3flapk.streamlit.app/

## 📌 Project Overview

The goal of this project is to build a text classification system that can identify patterns commonly associated with real and fake news articles.

The user can enter a news headline or article into the web application, and the trained machine learning model provides a prediction along with model probabilities.

> **Important:** This project is a machine learning classifier, not an independent fact-checking system. The model learns patterns from the training dataset and does not verify whether a claim is factually true.

## 📊 Dataset

The project uses the **ISOT Fake News Dataset**, containing approximately 45,000 news articles.

The dataset contains two categories:

- **Real News:** 21,417 articles
- **Fake News:** 23,481 articles
- **Total:** 44,898 articles

The main columns include:

- `title`
- `text`
- `subject`
- `date`

For this project, the `title` and `text` fields were combined to create the input text.

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Text Cleaning
   ↓
Train/Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Error Analysis
   ↓
Save Trained Model
   ↓
Streamlit Web Application

🧹 Text Preprocessing

The text was cleaned using simple NLP preprocessing techniques:

Converted text to lowercase
Removed URLs
Removed special characters and numbers
Removed extra whitespace
Combined article title and body text
🔢 TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the news articles into numerical features that machine learning models can understand.

The vectorizer was configured with:

Maximum features: 50,000
English stop-word removal

The TF-IDF vectorizer was fitted only on the training data to avoid data leakage.

🤖 Machine Learning Models

Three machine learning models were evaluated:

1. Logistic Regression

Accuracy:

99.00%

2. Linear SVM

Accuracy:

99.61%

3. Multinomial Naive Bayes

Accuracy:

Approximately 95%

Linear SVM achieved the highest test accuracy among the three models.

📈 Model Comparison
Model	Accuracy
Logistic Regression	99.00%
Linear SVM	99.61%
Naive Bayes	~95%
🔍 Linear SVM Confusion Matrix

The Linear SVM produced the following confusion matrix on the test set:

[[4675   21]
 [  14 4270]]

This means:

4,675 fake articles were correctly classified as fake
21 fake articles were classified as real
14 real articles were classified as fake
4,270 real articles were correctly classified as real

The test set contained 8,980 articles.

🌐 Streamlit Application

The trained model and TF-IDF vectorizer were saved using Joblib.

The Streamlit application allows users to:

Enter a news headline or article
Click the Predict button
Receive a classification
View the model probabilities

Example output:

Prediction: POTENTIALLY MISLEADING

Potentially Misleading: 98.09%
Real: 1.91%
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Logistic Regression
Multinomial Naive Bayes
Linear SVM
Matplotlib
Seaborn
Joblib
Streamlit
GitHub
📁 Project Structure
fake-news-detection-nlp/
│
├── app.py
├── fake_news_svm.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
▶️ Run Locally

Clone the repository:

git clone https://github.com/Piyush-Badlia/fake-news-detection-nlp.git

Move into the project directory:

cd fake-news-detection-nlp

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py
⚠️ Limitations

This model should not be treated as a factual verification system.

The classifier learns statistical and linguistic patterns present in the training dataset. Therefore:

A high model probability does not prove that an article is false.
The model may learn dataset-specific patterns.
Performance on real-world news may differ from performance on the test dataset.
Articles from topics or sources that differ from the training data may be classified incorrectly.

Therefore, the application uses the term "Potentially Misleading" rather than claiming that an article is definitively fake.

👨‍💻 Author

Piyush Badlia

M.Sc. Computer Science

GitHub: https://github.com/Piyush-Badlia


### Step 2: Commit the README

After pasting it:

1. Scroll to the bottom.
2. Find **Commit changes**.
3. Commit message:

```text
Add project README
Click Commit changes.
