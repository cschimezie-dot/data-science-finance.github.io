---
layout: post
title: "Don’t Ghost The Broker: Borrower Fallout Prediction"
image: "/posts/borrower_fallout_title_image.jpg"
tags: [Python, Pandas, Machine Learning, Random Forest, Commercial Lending, Lead Scoring]
---

# "Don’t Ghost The Broker" Borrower Fallout Prediction

## Project Overview

This project focused on a real business problem for Noble Equity, which answers:

** "Which commercial mortgage borrowers are likely to pay the application fee, and which borrowers are likely to disappear before moving forward based on previous application fee payments?" ****

In commercial lending, lead volume alone is not enough. A company can receive many borrower applications, but if those borrowers do not pay, submit documents, or continue through the process, the sales team wastes time chasing weak leads. It was very frustrating for mortgage sales reps to not go home without a paycheck.

The goal of this project was to build a borrower-fallout prediction model that helps Noble Equity distinguish serious borrowers from non-serious borrowers, and develop a user-friendly web application so stakeholders and clients can use and vet borrowers in their free time, saving the tech headaches and the time required to open a computer. 

In simple terms, the model helps answer the sales team:

> "Which borrowers are worth follow-up, and which borrowers are more likely to ghost the broker?"

Answers to stakeholders:

> "How should we position our next steps, and KPI's? and marketing reallocation "

---

## Business Problem

Noble Equity had borrower application data and payment data. The application data showed who submitted loan information. The payment data showed who actually paid the application fee.

The business issue was that 80% of borrowers began the process but did not take the next serious step: making a payment.

<img width="386" height="88" alt="Screenshot 2026-07-09 at 5 19 27 PM" src="https://github.com/user-attachments/assets/af910883-ec9f-4dbd-9a11-5b08572d6165" />


That created a clear business problem:

* Some borrowers apply but do not pay
* Sales time gets wasted on weak leads
* Strong borrowers are harder to prioritize
* Our business decided a better way to rank lead quality
* The team needs a repeatable system for spotting dropout risk

This project turned that problem into a machine learning classification task.

---

## Target Variable

I created a target column called:

```text
dropped_out
```

The meaning was simple:

* `0` = borrower paid the application fee
* `1` = borrower dropped out and did not pay

This gave the model a clear answer to learn from.

The model was trained to predict whether a borrower was more likely to drop out based on borrower and loan application details.

---

## Data Used

The project used:

* Noble Equity borrower application data
* Noble Equity payment data
* Borrower emails to connect applications to payment outcomes

The main borrower features included, despite there being over 30:

* Requested loan amount
* Credit score
* Liquidity
* Annual income
* Loan type
* Financing type
* Property type
* State

These are practical business fields because they are already part of the borrower intake process.

---

## Data Cleaning

A major part of this project was cleaning the borrower data.

The raw application data had messy values such as:

* Removed Dollar signs
* Merged data sets from different application software. 
* Commas
* Outlier Handling
* Text inside money fields
* Missing values
* Unknown values
* Monthly income formats
* Unrealistic outliers
* Different spellings and formats for categories
* Used Codex CL to optimize the dataset further by standardizing categorical variables, like changing state classification to region, for better prediction.
* Changing all the loan types from Primary and Unknown to Investment Only
* Standardized the property types to a few categories; for example, Single F to Single Family or unique property types like RV park to Unique Property Types
  

I cleaned and encoded the key numeric fields so the model could read them as numbers:

* `requested_loan_amount`
* `credit_score`
* `liquidity`
* `annual_income`

I also used KNN imputation to fill in missing numeric values by using similar borrowers, since the dataset of paid borrowers was too small to remove. It would hurt the data. On the Data Frames, you'll spot all the encoded columns.

This matters because a machine learning model cannot reliably predict from messy borrower responses.

---

## Outlier Handling

I chose not to remove them but to fill them with K-nearest neighbors, or KNN. I did not want to reduce the dataset's quantity; we are a small company. The liquidity column had some very large outliers. If those values were left untouched, the model could treat those extreme borrowers as more important than they really were.

To address that, I decided to create a safer modeling column called liquidity_capped so the model would know where we draw the line. This resulted in safer predictions for the model. 

```text
liquidity_capped
```

The original liquidity column stayed in the dataset for reference, but the model used `liquidity_capped` because it was safer for prediction.

---

## Tools Used

* Python
* Pandas
* Scikit-learn
* OneHotEncoder
* KNNImputer
* Random Forest Classification
* Train/Test Split
* Feature Importance
* Confusion Matrix 
* Joblib model saving

---

## Simplified Python Workflow

```python
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

```python
# Define model features

model_numeric_features = [
    "requested_loan_amount",
    "credit_score",
    "liquidity_capped",
    "annual_income"
]

model_categorical_features = [
    "loan_type",
    "financing_type",
    "property_type",
    "state"
]

X = borrower_analysis_clean[
    model_numeric_features + model_categorical_features
].copy()

y = borrower_analysis_clean["dropped_out"]
```

```python
# One-hot encode text columns

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded_array = one_hot_encoder.fit_transform(
    X[model_categorical_features]
)

encoded_columns = one_hot_encoder.get_feature_names_out(
    model_categorical_features
)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoded_columns,
    index=X.index
)

X_model = pd.concat(
    [
        X[model_numeric_features],
        encoded_df
    ],
    axis=1
)
```

```python
# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

```python
# Train Random Forest Classification model

dropout_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

dropout_model.fit(X_train, y_train)
```

```python
# Predict dropout and dropout probability

y_pred = dropout_model.predict(X_test)

y_pred_probability = dropout_model.predict_proba(X_test)[:, 1]
```

```python
# Evaluate model

print("Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

```python
# Feature importance

feature_importance_report = pd.DataFrame({
    "feature": X_model.columns,
    "importance": dropout_model.feature_importances_
}).sort_values(
    by="importance",
    ascending=False
)

feature_importance_report.head(25)
```

```python
# Save full model package

noble_equity_model_package = {
    "dropout_model": dropout_model,
    "one_hot_encoder": one_hot_encoder,
    "model_columns": X_model.columns.tolist(),
    "liquidity_cap": liquidity_cap,
    "model_numeric_features": model_numeric_features,
    "model_categorical_features": model_categorical_features
}

joblib.dump(
    noble_equity_model_package,
    "noble_equity_dropout_model_package.pkl"
)
```

---

## Key Discoveries
## Before AI Enhancement
<img width="826" height="205" alt="Screenshot 2026-07-08 at 3 15 28 PM" src="https://github.com/user-attachments/assets/aebe7087-8b52-447d-87c1-195bcd897c66" />


## After AI Enhancement


<img width="600" height="239" alt="Screenshot 2026-07-09 at 4 48 46 PM" src="https://github.com/user-attachments/assets/cf50b6c0-14ed-4e7e-b2cf-e27303f1491a" />

So after running everything. The first model found that loan amount, credit score, liquidity, and annual income accounted for 40% of borrower dropouts. 12% for the requested loan amount, 10.80% for credit, 10.81% for liquidity, and 10.13% for income. This model achieved an accuracy of about 85% using the F1 score and a confusion matrix. See below; I left that there so you can see. I also ran another model with AI to optimize more items that I thought could have influenced the prediction results. Because I did get alot of overfitting, meaning data leaks, I wanted to make sure the model was properly rinsed.  

Final Training accuracy: 0.9474
Final Testing accuracy: 0.8596
Accuracy gap: 0.0878

## Feature Importance Report Before AI Enhancement ( Model 1)
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        15
           1       0.90      1.00      0.95       129

    accuracy                           0.90       144
   macro avg       0.45      0.50      0.47       144
weighted avg       0.80      0.90      0.85       144

## Feature Importance Report After AI Enhancement ( Model 2)
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        19
           1       0.89      0.96      0.92       159

    accuracy                           0.86       178
   macro avg       0.44      0.48      0.46       178
weighted avg       0.79      0.86      0.83       178

Final Training accuracy: 0.9474
Testing accuracy: 0.8596
Accuracy gap: 0.0878


## Confusion Matrix Model 1
<img width="1042" height="849" alt="image" src="https://github.com/user-attachments/assets/d56a1aa1-7a56-4e32-ba9b-4f3d4b224942" />

## Confusion Matrix Model 2 (Unable to get this to Print)
[[  0  19]
 [  6 153]]
[0, 0] = Actual Paid, Predicted Paid
[0, 1] = Actual Paid, Predicted Dropped Out
[1, 0] = Actual Dropped Out, Predicted Paid
[1, 1] = Actual Dropped Out, Predicted Dropped Out

This means that instead of guessing which leads are serious, Noble Equity can assign a reliable predictive rating or score next to each borrower lead as it enters their pipeline, so they can prioritize their leads! (See data Frame below) 

<img width="733" height="187" alt="Screenshot 2026-07-09 at 5 57 04 PM" src="https://github.com/user-attachments/assets/40e6546f-b4a3-47d6-86b8-a63ecdef6943" />

---

## Business Output

The final output gave each borrower:

* A predicted dropout result
* A dropout probability percentage
* A risk bucket
* The cleaned borrower profile
* Key model features used for prediction

The risk buckets were:

* Low Dropout Risk
* Medium Dropout Risk
* High Dropout Risk

This makes the model easier for a business team to use.

---

## Business Impact

This model helps Noble Equity stop treating every borrower lead the same.

Instead of spending equal time on every application, the company can use dropout probability to prioritize follow-up.

##The model can help the business:

* Focus on stronger borrower leads
* Reduce wasted follow-up time by up to 30 hours per week
* Identify weak leads earlier
* Improve sales team efficiency with a web application making quick predictions for borrowers found outside the pipeline.
* Improve borrower pipeline quality by improving loan application questions; this one was huge. 
* Support better marketing and intake decisions for moving forward.
* Start implementing this tool for marketing for new leads as well as LOS systems so the team can focus on which leads are likely to close in order to meet Quota. The possibilities are endless!

The main value is that Noble Equity can use data to decide which borrowers deserve more attention at every stage of the borrower process.

---

## Final Summary and Limitations 

This project built a Random Forest Classification model to predict borrower fallout for Noble Equity.

The model connected borrower application data to payment outcomes, cleaned messy borrower fields, handled missing values and outliers, and created dropout probability scores. However, we may not have had enough borrowers to obtain an accurate sample of the full population we intended to study. The best we could do was bootstrap the data, just duplicating what we already had. Several lending firms process thousands of loans each month, while a small brokerage like Noble Equity was limited in the number of clients, loan officers, and marketing resources available to collect sufficient data. 

In plain English, this project helps Noble Equity answer:

> "Which borrowers are serious, and which borrowers are likely to disappear before paying?"

That makes the project useful for lead scoring, borrower prioritization, sales follow-up, and commercial mortgage pipeline management.
We will take further action to increase marketing to our ideal clients so we can collect more data and stress-test the model in other avenues.



---

## User-friendly web application!

Borrower Fallout Prediction a user-friendly form that one can enter in and obtain a high or low probability of a client dropping out. 

## Medium Probability Of Dropout!
<img width="630" height="1092" alt="image" src="https://github.com/user-attachments/assets/8e606036-5c3a-463e-8bfd-e1168601a4b5" />
## High Probability of Dropout!
<img width="657" height="1028" alt="image" src="https://github.com/user-attachments/assets/6b22f1b2-bc72-43e1-a98f-51dd24827d99" />
