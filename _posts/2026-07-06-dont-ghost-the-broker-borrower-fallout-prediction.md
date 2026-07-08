---
layout: post
title: "Don’t Ghost The Broker: Borrower Fallout Prediction"
image: "/posts/borrower_fallout_title_image.jpg"
tags: [Python, Pandas, Machine Learning, Random Forest, Commercial Lending, Lead Scoring]
---

# "Don’t Ghost The Broker" Borrower Fallout Prediction

## Project Overview

This project focused on a real business problem for Noble Equity:

**Which commercial mortgage borrowers are likely to pay the application fee, and which borrowers are likely to disappear before moving forward?**

In commercial lending, lead volume alone is not enough. A company can receive many borrower applications, but if those borrowers do not pay, submit documents, or continue through the process, the sales team wastes time chasing weak leads. It was very frustrating for mortgage sales reps to not go home without a paycheck.

The goal of this project was to build a borrower-fallout prediction model that helps Noble Equity distinguish serious borrowers from those more likely to drop out. 

In simple terms, the model helps answer:

> "Which borrowers are worth follow-up, and which borrowers are more likely to ghost the broker?"

---

## Business Problem

Noble Equity had borrower application data and payment data. The application data showed who submitted loan information. The payment data showed who actually paid the application fee.

The business issue was that many borrowers began the process but did not take the next serious step: making a payment.

That created a clear business problem:

* Some borrowers apply but do not pay
* Sales time gets wasted on weak leads
* Strong borrowers are harder to prioritize
* The business needs a better way to rank lead quality
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
* Text inside money fields
* Missing values
* Unknown values
* Monthly income formats
* Unrealistic outliers
* Different spellings and formats for categories

I cleaned the key numeric fields so the model could read them as numbers:

* `requested_loan_amount`
* `credit_score`
* `liquidity`
* `annual_income`

I also used KNN imputation to fill in missing numeric values by using similar borrowers, since the dataset of paid borrowers was too small to remove. It would hurt the data.

This matters because a machine learning model cannot reliably predict from messy borrower responses.

---

## Outlier Handling

The liquidity column had some very large outliers. If those values were left untouched, the model could treat those extreme borrowers as more important than they really were.

To solve that, I created a safer modeling column:

```text
liquidity_capped
```

This kept normal liquidity values the same but limited extreme values above the cap.

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

The model demonstrated that borrower fallout can be analyzed by examining patterns in application data. Testing found that loan amount and credit score were t Liquidity, annual income, and years in business each accounted for 40.03% of borrower fallouts. 9.72% for requested loan amount, 8.35% for credit, 8.38% for liquidity, 8.27% for income, and 6.86% for years in business. Our model achieved an accuracy of 85% using the F1 score metric and confusion matrix. See below.

## Feature Importance Report
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        15
           1       0.90      1.00      0.95       129

    accuracy                           0.90       144
   macro avg       0.45      0.50      0.47       144
weighted avg       0.80      0.90      0.85       144

<img width="1042" height="849" alt="image" src="https://github.com/user-attachments/assets/d56a1aa1-7a56-4e32-ba9b-4f3d4b224942" />

This means instead of guessing which leads are serious, Noble Equity can use an accurate predictor model to estimate dropout risk.

The model used borrower financial strength, deal type, property type, geography, and financing request details to also factor in a dropout probability score.

A dropout probability is more useful than a simple yes-or-no answer.

For example:

```text
82% dropout probability
```

means the borrower has a high estimated chance of not moving forward.

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

The model can help the business:

* Focus on stronger borrower leads
* Reduce wasted follow-up time
* Identify weak leads earlier
* Improve sales team efficiency
* Improve borrower pipeline quality
* Support better marketing and intake decisions

The main value is that Noble Equity can use data to decide which borrowers deserve more attention.

---

## Final Summary

This project built a Random Forest Classification model to predict borrower fallout for Noble Equity.

The model connected borrower application data to payment outcomes, cleaned messy borrower fields, handled missing values and outliers, and created dropout probability scores.

In plain English, this project helps Noble Equity answer:

> "Which borrowers are serious, and which borrowers are likely to disappear before paying?"

That makes the project useful for lead scoring, borrower prioritization, sales follow-up, and commercial mortgage pipeline management.
We will take further action in focusing 




---

## Portfolio Image

![Borrower Fallout Prediction](/img/posts/borrower_fallout_title_image.jpg "Don’t Ghost The Broker - Borrower Fallout Prediction")
