---
layout: post
title: "Catch The Bad Apples: Commercial Loan Vintage Risk Analysis"
image: "/posts/apples_in_a_wooden_bowl_still_life.jpg"
tags: [Python, Pandas, Machine Learning, Random Forest, Commercial Lending, Credit Risk, Vintage Analysis]
---

In this project, I built a commercial lending risk analytics workflow to predict net loss percentage and identify weak commercial loan vintages before losses become harder to control.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Business Problem](#business-problem)
- [02. Dataset Used](#dataset-used)
- [03. Tools Used](#tools-used)
- [04. Data Cleaning and Preparation](#data-cleaning)
- [05. Model Building Process](#model-building)
- [06. Model Validation](#model-validation)
- [07. Feature Importance Findings](#feature-importance)
- [08. Actual vs Predicted Results](#actual-vs-predicted)
- [09. Vintage Performance Findings](#vintage-performance)
- [10. Business Impact](#business-impact)
- [11. Final Business Summary](#final-summary)
- [12. What I Would Improve Next](#next-steps)

___

# Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Commercial lenders need to understand which loan batches are starting to weaken. A single loan may not tell the full story, but a loan vintage can show whether a group of loans originated around the same time is performing worse than expected.

This project uses synthetic commercial lending data to predict `Net_Loss_Pct`, a continuous loss percentage field. Because the target is numeric, I used a **Random Forest Regression** model. This was one of my first projects, so I decided to test it out and see how it did.

<br>

### Actions <a name="overview-actions"></a>

I cleaned the loan data, removed ID and date columns that could confuse the model, encoded the industry field, trained a Random Forest Regression model, and created validation checks to compare training and testing performance.

The workflow also produced:

* Model results summary
* Feature importance table
* Permutation importance table
* Actual vs predicted chart
* Regression loss-bucket matrix
* Vintage performance summary
* Months-on-book loss summary
* Saved model file

<br>

### Results <a name="overview-results"></a>

The model achieved a strong test R² score of `0.9630` and an average cross-validation score of `0.9611`. This is fairly normal for an AI-generated model; it tends to bootstrap from other existing datasets.

| Metric | Score |
|---|---:|
| R² Score | 0.9630 | ✔
| Mean Absolute Error (%) | 0.2665 | ✔
| Root Mean Squared Error (%) | 0.7381 | ✔
| Average Cross-Validation Score | 0.9611 |

The basic overfitting check showed:

| Metric | Score |
|---|---:|
| Training R² Score | 0.9945 |
| Testing R² Score | 0.9630 |
| Difference | 0.0315 | ✔

The training and testing scores are close, so the model appears stable and is performing well on new data, this is a basic overfitting check. However, the score remains very high, so I would review the data for leakage before treating it as a real credit model.

<br>

### Growth/Next Steps <a name="overview-growth"></a>

This project is portfolio-grade, not production-ready. The next step would be to test the workflow with real commercial lending data, stronger time-based validation, delinquency history, collateral fields, macroeconomic variables, and a deeper review of leakage, which is what working with an institution allows me to do.

___

# Business Problem <a name="business-problem"></a>

Commercial lenders can lose money when weak borrowers or weak loan batches are found too late. Revenue declines, credit utilization spikes, seasoning, and borrower risk ratings can all signal stress before losses become obvious.

The business problem was:

> Can we predict commercial loan net loss percentage and identify the loan vintages or borrower signals connected to higher loss risk?

The goal was not just to build a model. The goal was to create a risk analytics workflow that helps a credit team ask better questions about portfolio performance.

___

# Dataset Used <a name="dataset-used"></a>

The project uses `commercial_vintage_master_dataset.xlsx`, a synthetic commercial lending dataset made by AI with loan snapshot records.

The dataset includes:

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

The target variable was:

```text
Net_Loss_Pct
```

This is a continuous value, which is why regression was the correct modeling approach.

___

# Tools Used <a name="tools-used"></a>

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Random Forest Regression
* Cross-validation
* Feature importance
* Permutation importance
* Pickle model saving

___

# Data Cleaning and Preparation <a name="data-cleaning"></a>

The main preparation steps were:

1. Loaded the commercial loan vintage dataset
2. Checked rows, columns, data types, and missing values
3. Removed missing records
4. Checked outliers in loan amount, current balance, and annual revenue
5. Removed leakage and identifier columns
6. One-hot encoded `Industry_Sector`
7. Split the data into training and testing sets

The leakage columns removed from the model features were:

* `Net_Loss_Pct`
* `Snapshot_Date`
* `Origination_Date`
* `Loan_ID`
* `Borrower_ID`

This keeps the model focused on business signals rather than IDs, raw dates, or the answer column.

___

# Model Building Process <a name="model-building"></a>

The final model was a Random Forest Regression model.

```python
# Import libraries
  import pandas as pd
  import numpy as np

  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

  # Load dataset
  df = pd.read_excel("data/raw/commercial_vintage_master_dataset.xlsx")

  # Clean missing values
  df.dropna(inplace=True)

  # Set target variable
  y = df["Net_Loss_Pct"]

  # Remove target, IDs, and possible leakage columns
  X = df.drop(
      [
          "Net_Loss_Pct",
          "Snapshot_Date",
          "Origination_Date",
          "Loan_ID",
          "Borrower_ID"
      ],
      axis=1
  )

  # Convert industry text into number columns
  X = pd.get_dummies(
      X,
      columns=["Industry_Sector"],
      drop_first=True
  )

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=0.2,
      random_state=42
  )

  # Train Random Forest Regression model
  rf_model = RandomForestRegressor(
      random_state=42
  )

  rf_model.fit(X_train, y_train)

  # Predict Net_Loss_Pct
  y_pred = rf_model.predict(X_test)

  # Check model performance
  training_r2 = rf_model.score(X_train, y_train)
  testing_r2 = rf_model.score(X_test, y_test)
  r2 = r2_score(y_test, y_pred)
  mae_percent = mean_absolute_error(y_test, y_pred) * 100
  rmse_percent = np.sqrt(mean_squared_error(y_test, y_pred)) * 100

  cv_scores = cross_val_score(
      rf_model,
      X_train,
      y_train,
      cv=5,
      scoring="r2"
  )

  # Print results
  print("Training R2 Score:", training_r2)
  print("Testing R2 Score:", testing_r2)
  print("R2 Score:", r2)
  print("Mean Absolute Error (%):", mae_percent)
  print("Root Mean Squared Error (%):", rmse_percent)
  print("Average Cross-Validation Score:", cv_scores.mean())

  # Simple overfitting check
  if training_r2 - testing_r2 > 0.10:
      print("Warning: The model may be overfitting.")
  else:
      print("The model looks stable.")

```

I kept the model simple on purpose. I did not add GridSearchCV, advanced pipelines, or classification logic because the goal was a clear, beginner-friendly regression workflow for `Net_Loss_Pct`.

___

# Model Validation <a name="model-validation"></a>

The model was evaluated with:

* Training R² score
* Testing R² score
* R² score
* Mean absolute error
* Root mean squared error
* Cross-validation score

The model results were:

| Metric | Score |
|---|---:|
| R² Score | 0.9630 | 
| Mean Absolute Error (%) | 0.2665 |
| Root Mean Squared Error (%) | 0.7381 |
| Average Cross-Validation Score | 0.9611 |

The overfitting check compared training and testing R²:

* Training R² Score: `0.9945`
* Testing R² Score: `0.9630`
* Difference: `0.0315`

Because the gap is not large, the model appears stable based on a simple overfitting check. Still, the performance is unusually strong, so a deeper review of leakage would be important before using this type of model in a real-world credit environment. Withthe R² being so high technically, it means that the model predicted 96% of the reason why a net loss occurred. Think of R like how much the model "Revealed".

___

# Feature Importance Findings <a name="feature-importance"></a>

The strongest Random Forest feature importance signals were:

* `Credit_Utilization_Spike`
* `Original_Internal_Risk_Rating`
* `Revenue_Drop_20Pct`
* `Months_On_Book`

These signals make business sense. Borrowers with rising credit usage, weaker revenue, higher internal risk ratings, and longer seasoning can show higher loss risk.

<br>

![Feature Importance]

<img width="1200" height="630" alt="feature_importance_chart" src="https://github.com/user-attachments/assets/9ad1fb00-99d6-44fa-acfa-cc0c889cac81" />


___

# Actual vs Predicted Results <a name="actual-vs-predicted"></a>

The actual vs predicted chart compares the real `Net_Loss_Pct` values against the model predictions. Points close to the red diagonal line are better predictions.

<br>

![Actual vs Predicted Net Loss]

<img width="1200" height="630" alt="actual_vs_predicted_net_loss" src="https://github.com/user-attachments/assets/29a76046-ea90-480b-a714-5a661648e132" />


I also created a regression loss-bucket matrix. Since this is not a classification model, I first grouped actual and predicted values into low, medium, and high loss risk buckets, then compared those buckets.

<br>

![Loss Bucket Matrix]


<img width="1200" height="630" alt="loss_bucket_matrix" src="https://github.com/user-attachments/assets/d98ea0cd-6769-4252-a45b-d4f8198cbc12" />

___

# Vintage Performance Findings <a name="vintage-performance"></a>

The highest average-loss vintages in this run included:

* `2024-08-01`
* `2023-06-01`
* `2024-04-01`

These vintages would be good candidates for deeper review by a commercial credit or portfolio monitoring team.

<br>

![Vintage Performance]


<img width="1200" height="630" alt="vintage_performance_chart" src="https://github.com/user-attachments/assets/4483c28a-b872-4947-b460-63fa68cdfe46" />


Months-on-book analysis also showed that losses were not perfectly flat across loan age. This supports the value of monitoring seasoning patterns over time.

<br>

![Months On Book Loss]


<img width="1200" height="630" alt="months_on_book_loss_chart" src="https://github.com/user-attachments/assets/eb77bf26-c53d-4646-b3ce-ed98dcca9771" />

___

# Business Impact <a name="business-impact"></a>

This project shows how commercial lenders could use a regression model and vintage analysis to support portfolio monitoring.

The workflow can help a credit team:

* Identify weaker loan vintages
* Understand which borrower signals are driving loss risk
* Focus reviews on loans with stronger warning signs
* Monitor portfolio seasoning
* Improve underwriting feedback loops

The model should not be treated as production-ready. It is a polished portfolio project that demonstrates credit risk analytics, regression modeling, and business interpretation.

___

# Final Business Summary <a name="final-summary"></a>

This project built a Random Forest Regression model to predict commercial loan `Net_Loss_Pct`. The strongest signals were credit utilization spikes, original internal risk rating, revenue drops, and months on book.

In business terms, the model helps identify the bad apples in a commercial loan portfolio earlier. A lender could use this type of workflow to review risky vintages, investigate borrower stress signals, and improve portfolio monitoring before losses become larger.

___

# What I Would Improve Next <a name="next-steps"></a>

If I expanded this project, I would add:

1. Real commercial lending data
2. Time-based validation
3. Delinquency and payment history
4. Collateral and loan purpose details
5. Macroeconomic indicators
6. A dashboard for vintage monitoring
7. A deeper leakage review with credit risk stakeholders

These improvements would make the analysis more realistic and more useful for a real commercial lending environment.

 # Streamlit App Demo
 
 Local Hosting: http://localhost:8504/

  To make this project more interactive, I also built a local Streamlit app for the Commercial Loan Vintage Risk
  Analysis model. If you cannot click the link, see the image below!

  The app allows a user to enter a commercial loan profile and receive a predicted Net Loss Percentage from the saved
  Random Forest Regression model. The app does not retrain the model. It loads the saved model file and model column
  file, formats the user inputs into the same structure used during training, and then returns a single prediction.

  The app includes inputs for:

  - Months on book
  - Current balance
  - Original loan amount
  - Interest rate
  - Annual revenue
  - Original internal risk rating
  - Industry sector
  - Credit utilization spike
  - Revenue drop indicator

  After the user submits the form, the app displays the predicted net loss percentage and assigns the loan profile to a
  simple risk band:

   Risk Band           Predicted Net Loss %    Recommended Action
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Low Loss Risk                Below 1.00%    Continue normal monitoring
  ──────────────────  ──────────────────────  ──────────────────────────────────────────────────────────────────────────
   Medium Loss Risk          1.00% to 2.50%    Review borrower trends, revenue pressure, utilization changes, and
                                               vintage performance
  ──────────────────  ──────────────────────  ──────────────────────────────────────────────────────────────────────────
   High Loss Risk               Above 2.50%    Flag for credit review and prioritize portfolio monitoring

  This turns the project from a static machine learning analysis into a simple business-facing risk tool! A credit
  analyst or portfolio manager could use this type of app to quickly test loan profiles, understand estimated loss risk,
  and decide whether a loan should remain in normal monitoring or receive closer review.

  The Streamlit app is still a portfolio demo and is not intended to be a production credit decisioning tool. However,
  it shows how the model could be packaged into a practical interface for commercial lending analytics teams. Looking forward to hearing from you!

 # Streamlit App View

  <img width="752" height="664" alt="Screenshot 2026-07-10 at 12 02 30 PM" src="https://github.com/user-attachments/assets/67a4ac47-6ea2-43d3-a9e9-0607fb7b1bcf" />


