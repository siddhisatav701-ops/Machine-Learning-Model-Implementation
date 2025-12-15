# Machine-Learning-Model-Implementation
COMPANY: CODTECH IT SOLUTIONS

NAME: SIDDHI TANAJI SATAV

INTERN ID: CTIS0158

DOMAIN: Python Programming

DURATION: 4 WEEEKS

MENTOR: NEELA SANTOSH

Internship Project Submission

CODTECH Internship – Task 4
Title: Spam Message Classification Using Scikit‑Learn

1. Objective
Create a predictive model using scikit‑learn to classify text messages as spam or ham (not spam). The implementation and evaluation are presented in a Jupyter notebook, fulfilling the requirement to build and document a predictive model.​

2. Tech Stack
Language: Python 3
Environment: Jupyter Notebook (.ipynb)
Libraries:
pandas – data loading and preprocessing
scikit-learn – machine learning (model, train–test split, metrics)​
numpy – numerical support (used by scikit‑learn)
Install dependencies:
bash
pip install pandas scikit-learn numpy

3. Dataset
File: spam.csv (placed in the same folder as the notebook).
Structure:
text
label,message
spam,"Win a free iPhone now! Click this link to claim your prize."
ham,"Are we still meeting for lunch tomorrow at 1 pm?"
...
label – target variable (spam or ham).
message – SMS/email‑like text content.
This small dataset is enough to demonstrate the full ML workflow; it can be replaced with a larger public spam dataset if needed.​

4. Notebook Workflow
Filename: e.g., Task4_SpamClassifier.ipynb

4.1. Data Loading and Exploration
python
import pandas as pd
df = pd.read_csv("spam.csv")
df.head()
df['label'].value_counts()
Loads data into a DataFrame.
Checks class distribution to see how many spam vs ham messages exist.

4.2. Train–Test Split
python
from sklearn.model_selection import train_test_split
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Splits the dataset into training and test sets to evaluate generalization.​

4.3. Text Vectorization (TF‑IDF)
python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
Converts raw text into numerical feature vectors using TF‑IDF, a common technique in spam detection.​

4.4. Model Training
python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_vec, y_train)
Uses Multinomial Naive Bayes, a standard algorithm for text classification, especially spam filters.​

4.5. Evaluation
python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
Reports accuracy, confusion matrix, precision, recall, and F1‑score for spam and ham.
In markdown cells, the notebook explains what these metrics mean and comments on model performance.
​
4.6. Predicting New Messages
python
def predict_message(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

print(predict_message("Congratulations! You have won a free lottery. Click here."))
print(predict_message("Please send the project files by evening."))
Demonstrates how to use the trained model for new, unseen messages.
Shows example outputs to illustrate practical use.

5. How to Run
Ensure the following files are in the same folder:
Task4_SpamClassifier.ipynb
spam.csv
Install dependencies (once):
bash
pip install pandas scikit-learn numpy
Launch Jupyter:
bash
jupyter notebook
Open Task4_SpamClassifier.ipynb and run all cells in order (Kernel → Restart & Run All).
You should see:
Data preview and label counts.
Successful model training.
Accuracy and other metrics printed.
Example predictions for custom messages.

6. Deliverable Summary
Jupyter Notebook: complete implementation and evaluation of the spam classifier.
Dataset: spam.csv used for training and testing.
The notebook clearly demonstrates the end‑to‑end ML pipeline: data loading, preprocessing, model building, evaluation, and prediction on new data using scikit‑learn.

Output


<img width="418" height="216" alt="image" src="https://github.com/user-attachments/assets/fc3d26d3-20a0-44b1-9940-0902f793d601" />
