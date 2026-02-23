🚢 Titanic Survival Prediction – End-to-End Machine Learning Pipeline
📌 Project Overview

This project implements a complete machine learning pipeline to predict passenger survival using the Titanic dataset.

The pipeline covers:

Data Cleaning & Missing Value Handling

Feature Engineering

Exploratory Data Analysis (EDA)

One-Hot Encoding

Train-Test Splitting

Model Training (Random Forest)

Performance Evaluation

Visualization


The goal was to build a modular, production-style ML project structure, not just a notebook experiment.

📊 Dataset

Dataset: Titanic dataset (Kaggle)

Key features used:

Pclass – Passenger Class

Sex – Gender

Age – Passenger Age

Fare – Ticket Fare

Embarked – Port of Embarkation


Target variable:

Survived (0 = No, 1 = Yes)

🛠️ Tech Stack

Python

Pandas

NumPy

Matplotlib

Scikit-Learn


🤖 Model

Random Forest Classifier

Why Random Forest?

Handles non-linear relationships well

Robust to outliers

Less sensitive to multicollinearity

Provides feature importance

📈 Results

Accuracy: ~80%

Balanced Precision and Recall

Evaluated using:

Confusion Matrix

Classification Report


Feature Importance

Confusion Matrix Breakdown:

True Negatives: 88

True Positives: 55

False Positives: 17

False Negatives: 19

The model performs consistently across both survival classes.

📷 Visual Results
Confusion Matrix

Survival by Gender


🗂️ Project Structure
titanic-ml-project/
│
├── data/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── main.py
│
├── outputs/
├── requirements.txt
└── README.md

This structure follows modular design principles used in real-world ML projects.

🚀 How to Run

Clone the repository:

git clone https://github.com/Aarush-Pradhan/titanic-ml-project.git
cd titanic-ml-project

Install dependencies:

pip install -r requirements.txt

Run the pipeline:

python main.py
🎯 Key Learnings

Handling missing values strategically

Avoiding dummy variable trap with one-hot encoding

Interpreting confusion matrix mathematically

Evaluating classification models properly

Building structured ML projects (not just notebooks)

👨‍💻 Author

Aarush