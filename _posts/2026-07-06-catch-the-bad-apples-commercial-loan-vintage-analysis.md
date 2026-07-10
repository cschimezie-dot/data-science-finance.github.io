---
layout: post
title: "Catch The Bad Apples: Commercial Loan Vintage Risk Analysis"
image: "/posts/apples_in_a_wooden_bowl_still_life.jpg"
tags: [Python, Pandas, Machine Learning, Random Forest, Commercial Lending, Risk Analytics]
---

# "Catch The Bad Apples" Commercial Loan Vintage Risk Analysis

## Project Overview

This project focused on one simple business question:

**Which groups of commercial loans are starting to perform badly, and can we spot the warning signs earlier?**

In commercial lending, lenders do not care only about whether a loan is good or bad. They also care about **loan vintages**. A loan vintage is a group of loans that started around the same time. If one batch of loans begins showing higher losses than the others, the lender needs to know why.

For this project, I built a machine learning model to predict **Net Loss Percentage** from commercial loan records. The goal was to help a lender identify borrower, credit, and portfolio signals connected to higher loan losses.

In simple terms, this model helps answer:

> "Which loans look like bad apples before they spoil the whole portfolio?"

---

## Business Problem

Commercial lenders can lose money when weak loans are found too late. A borrower may look fine at origination, but later the loan can weaken due to revenue declines, higher credit utilization, seasoning, changes in borrower risk, or industry stress.

The business problem was to create a model that could help lenders:

* Detect weak loan batches earlier
* Identify borrower characteristics tied to higher losses
* Monitor portfolio risk across loan origination periods
* Reduce underwriting blind spots
* Improve credit monitoring before losses become larger

This is important because a lender does not want to wait until a loan fully defaults before noticing the warning signs.

---

## Dataset Used

The project used an AI-generated dataset of commercial loan vintages with loan- and borrower-level information.

The dataset included columns such as:

* `Loan_ID`
* `Borrower_ID`
* `Snapshot_Date`
* `Origination_Date`
* `Months_On_Book`
* `Loan_Amount`
* `Current_Balance`
* `Interest_Rate`
* `Industry_Sector`
* `Annual_Revenue`
* `Original_Internal_Risk_Rating`
* `Credit_Utilization_Spike`
* `Revenue_Drop_20Pct`
* `Net_Loss_Pct`

The target column was:

```text
Net_Loss_Pct
```

That means the model was trained to predict the percentage loss associated with each loan record.

---

## Tools Used

* Python
* ChatGPT 
* Pandas
* NumPy
* Scikit-learn
* Random Forest Regression
* Train/Test Split
* Cross Validation
* Feature Importance
* Joblib / Pickle-style model saving

---

## Data Cleaning and Preparation

Before training the model, I cleaned and prepared the data so the model could understand it properly.

##The main preparation steps were:

1. Loaded the AI-generated commercial loan vintage dataset
2. Checked rows, columns, and missing values
3. Removed ID columns that should not be used for prediction
4. Removed raw date columns that caused model errors
5. Converted categorical fields like `Industry_Sector` into numeric dummy variables
6. Split the data into `X` features and `y` target
7. Used a train/test split to test the model fairly
8. Model Validation to ensure accuracy. 

One important lesson from this project was that raw date and ID columns can break or mislead a model. The model should learn from business features, not from meaningless identifiers.

---

## Python Code

```python
# =============================================================================
# 1. Import Libraries
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib
```

```python
# =============================================================================
# 2. Load Dataset
# =============================================================================

commercial_vintage = pd.read_excel(
    "commercial_vintage_master_dataset.xlsx"
)

# Loads the commercial loan vintage dataset
```

```python
# =============================================================================
# 3. Inspect Dataset
# =============================================================================

commercial_vintage.head()

commercial_vintage.info()

commercial_vintage.isna().sum()

# Checks the structure, data types, and missing values
```

```python
# =============================================================================
# 4. Define Target
# =============================================================================

y = commercial_vintage["Net_Loss_Pct"]

# y is the value we want the model to predict
```

```python
# =============================================================================
# 5. Create X Features
# =============================================================================

X = commercial_vintage.drop(
    columns=[
        "Net_Loss_Pct",
        "Loan_ID",
        "Borrower_ID",
        "Snapshot_Date",
        "Origination_Date"
    ]
)

# Removes the target, IDs, and raw date columns
# Keeps business features the model can learn from
```

```python
# =============================================================================
# 6. One-Hot Encode Categorical Features
# =============================================================================

X = pd.get_dummies(
    X,
    columns=["Industry_Sector"],
    drop_first=True
)

# Converts industry sector text into numeric columns
```

```python
# =============================================================================
# 7. Train Test Split
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Splits data into training and testing sets
```

```python
# =============================================================================
# 8. Train Random Forest Regression Model
# =============================================================================

vintage_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

vintage_model.fit(X_train, y_train)

# Trains the model
```

```python
# =============================================================================
# 9. Make Predictions
# =============================================================================

y_pred = vintage_model.predict(X_test)

# Predicts net loss percentage on unseen test data
```

```python
# =============================================================================
# 10. Evaluate Model
# =============================================================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# R2 shows how much of the loss pattern the model explains
# MAE and RMSE show average prediction error
```

```python
# =============================================================================
# 11. Cross Validation
# =============================================================================

cv_scores = cross_val_score(
    vintage_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

# Tests the model across multiple splits of the data
```

```python
# =============================================================================
# 12. Feature Importance
# =============================================================================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": vintage_model.feature_importances_
}).sort_values(
    by="importance",
    ascending=False
)

feature_importance.head(20)

# Shows which features were most important to the model
```

```python
# =============================================================================
# 13. Save Model
# =============================================================================

joblib.dump(
    vintage_model,
    "commercial_loan_vintage_random_forest_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "commercial_loan_vintage_model_columns.pkl"
)

# Saves the model and column order for future use
```

---

## Key Discoveries

The model was useful because it showed that commercial loan losses are not random. Certain borrower and portfolio patterns can signal higher risk.

The most important warning signs to monitor were:

* **Revenue deterioration** - borrowers with major revenue drops were more likely to show loss pressure, possibly due to stressful situations.
* **Credit utilization spikes** - borrowers using more available credit may be under financial stress.
* **Loan seasoning** - loans can behave differently depending on how many months they have been active.
* **Current balance exposure** - larger remaining balances can create larger portfolio risk.
* **Internal risk rating** - higher-risk borrower ratings can help explain future loss behavior.
* **Industry sector** - some industries may show more weakness than others during stressed periods.

The main discovery was that lenders can use predictive analytics to identify weak loan batches before losses fully materialize.

---

## Business Impact

This project shows how a lender could use machine learning to improve commercial credit monitoring.

Instead of waiting for losses to appear, the lender could use the model to score loans earlier and flag risky vintages for review.

The model can support:

* Better portfolio monitoring
* Earlier risk detection
* Smarter credit review meetings
* More focused borrower follow-up
* Better underwriting feedback
* Stronger vintage performance analysis

The business value is that lenders can act earlier. If a batch of loans is showing warning signs, the credit team can investigate before the problem gets bigger.

---

## Final Business Summary

This project built a Random Forest Regression model to predict net loss percentage across commercial loan vintages. The model used borrower, loan, revenue, utilization, industry, and seasoning data to identify signals connected to weaker portfolio performance.

In plain English, the model helps lenders find the bad apples in the loan portfolio earlier.

That means the lender can spot risky batches, understand what is driving poor performance, and improve credit monitoring before losses become harder to control.

---

## What I Would Improve Next

If this project were expanded, I would add:

1. More borrower payment history
2. Macroeconomic indicators
3. Collateral/property-level data
4. Delinquency history
5. Vintage-level dashboard visuals
6. Probability bands for low, medium, and high loss risk
7. Accuracy scoring, using real data sets for better authenticity 

This would make the model even more useful for real-world portfolio management.

---

## Portfolio Image

![Commercial Loan Vintage Analysis](/img/posts/commercial_loan_vintage_bad_apples.png "Catch The Bad Apples - Commercial Loan Vintage Analysis")

---
