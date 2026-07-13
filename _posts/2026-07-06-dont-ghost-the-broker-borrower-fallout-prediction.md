---
layout: post
title: "Don't Ghost the Broker: Borrower Fallout Prediction"
image: "/posts/borrower_fallout_title_image.jpg"
tags: [Python, Machine Learning, Classification, Random Forest, Data Cleaning]
---

This project predicts whether a commercial mortgage borrower is likely to continue by paying the application fee or drop out after applying.

I built it in two parts. **395 Fall_Out_Part_1** created the target variable by matching application records to successful payment records. **395 Fall_Out_Part_2** improved the modeling table by standardizing messy business categories into cleaner, stakeholder-friendly groups.

![Borrower fallout hero](/img/posts/borrower_fallout_hero.png)

# Table of Contents

- [00. Project Overview](#overview-main)
  - [Context](#overview-context)
  - [Actions](#overview-actions)
  - [Results](#overview-results)
  - [Growth/Next Steps](#overview-growth)
- [01. Business Problem](#business-problem)
- [02. Target Variable](#target-variable)
- [03. Data Sources](#data-sources)
- [04. Data Lineage](#data-lineage)
- [05. Why This Required Two Parts](#two-parts)
- [06. 395 Fall_Out_Part_1](#part-1)
- [07. What Made Part 1 Difficult](#part-1-difficult)
- [08. Part 1 Results](#part-1-results)
- [09. Why Accuracy Was Misleading](#accuracy-misleading)
- [10. 395 Fall_Out_Part_2](#part-2)
- [11. Part 2 Results](#part-2-results)
- [12. Part 1 vs Part 2](#comparison)
- [13. Model Interpretation](#interpretation)
- [14. Probability And Risk Buckets](#probability-risk)
- [15. Bootstrapping Explanation](#bootstrapping)
- [16. Business Impact](#business-impact)
- [17. Web Application](#web-application)
- [18. Limitations](#limitations)
- [19. Lessons Learned](#lessons-learned)
- [20. Reproducibility](#reproducibility)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Noble Equity receives commercial mortgage applications from borrowers who may or may not pay the application fee. The business needed a way to identify applicants who are likely to continue and applicants who may need follow-up before they disappear from the pipeline.

### Actions <a name="overview-actions"></a>

I built a two-part Random Forest Classification workflow:

- Part 1 connected application records to successful payment records using normalized borrower emails.
- Part 1 created the `dropped_out` target because it did not exist in the raw application files.
- Part 1 trained a baseline model on the original borrower fields.
- Part 2 started from `Book3.xlsx`, then standardized states, financing types, and property types.
- Part 2 produced probability predictions, risk buckets, feature importance, confusion matrices, and stakeholder reports.

### Results <a name="overview-results"></a>

The project produced a working local proof-of-concept, but the results must be interpreted carefully.

- Part 1 testing accuracy: **89.58%**
- Part 2 testing accuracy: **85.96%**
- Part 1 model columns: **225**
- Part 2 model columns: **19**
- Part 2 reduced raw property categories from **83** to **3** and financing categories from **60** to **5**.

The most important limitation is that both held-out tests showed **0.00 paid-class recall**. The models identified most dropout cases, but they did not correctly identify paid borrowers in the test set.

### Growth/Next Steps <a name="overview-growth"></a>

The next improvement should focus on collecting more paid-borrower examples, validating on future borrower records, and testing business interventions before using this for live decisions.

___

# 01. Business Problem <a name="business-problem"></a>

The business question was:

> Which commercial mortgage borrowers are likely to continue by paying the application fee, and which applicants are likely to drop out?

This is a classification problem, not a regression problem, because the model predicts two classes:

- Paid
- Dropped out

___

# 02. Target Variable <a name="target-variable"></a>

The target variable is:

```text
dropped_out
```

Target meaning:

- `0` = borrower paid the application fee
- `1` = borrower applied but did not pay

The target was never used as a model feature. It was only used as the answer column.

___

# 03. Data Sources <a name="data-sources"></a>

Part 1 used three raw files:

- `NE_app_1.csv`
- `NE_App_2.csv`
- `Noble_Equity_Payments.csv`

Part 2 used:

- `Book3.xlsx`

The raw files include private borrower information, so I did not expose real borrower emails in this write-up. Example emails should be masked like:

```text
jo***@example.com
```

___

# 04. Data Lineage <a name="data-lineage"></a>

![Borrower fallout data lineage](/img/posts/borrower_fallout_data_lineage.png)

The key step was email matching. The application data showed who applied. The payment data showed who paid. Matching the two created the modeling target.

___

# 05. Why This Required Two Parts <a name="two-parts"></a>

Part 1 was necessary because the raw application files did not already contain the answer. I had to build the answer by connecting borrower applications to successful payment records.

Part 2 was necessary because the first model had too much category fragmentation. The same real-world category appeared under many labels, so one-hot encoding created too many columns and made the model harder to explain.

___

# 06. 395 Fall_Out_Part_1 <a name="part-1"></a>

Part 1 began with the raw application and payment files.

## Loading The Source Files

```python
app_1 = pd.read_csv(PROJECT_ROOT / "NE_app_1.csv")
app_2 = pd.read_csv(PROJECT_ROOT / "NE_App_2.csv")
payments = pd.read_csv(PROJECT_ROOT / "Noble_Equity_Payments.csv")
```

This step loaded the two application systems and the payment table.

## Cleaning Borrower Emails

```python
app_1["borrower_email"] = app_1["Email Address"].astype(str).str.lower().str.strip()
app_2["borrower_email"] = app_2["Username"].astype(str).str.lower().str.strip()
payments["payment_email"] = payments["Customer Email"].str.lower().str.strip()
```

App 1 used `Email Address`, while App 2 used `Username`. Normalizing both into `borrower_email` allowed the records to be compared fairly.

## Creating The Target

```python
paid_payments = payments[
    payments["Status"].astype(str).str.lower().str.contains("paid", na=False)
]

paid_emails = set(paid_payments["payment_email"].dropna())
applications["paid_application_fee"] = applications["borrower_email"].isin(paid_emails)

applications["dropped_out"] = applications["paid_application_fee"].map({
    True: 0,
    False: 1
})
```

This created the answer column. A borrower who matched a successful payment email was labeled as paid. Everyone else was labeled as dropped out.

## KNN Imputation

```python
knn_imputer = KNNImputer(n_neighbors=5)
knn_model_table_imputed_scaled_array = knn_imputer.fit_transform(
    knn_model_table_scaled
)
```

The original dataset was small, so deleting missing rows would have made the data even smaller. KNN imputation helped keep more borrower records while filling missing numeric values.

## Training Random Forest

```python
dropout_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

dropout_model.fit(X_train, y_train)
```

Random Forest Classification was used because the target is paid versus unpaid.

___

# 07. What Made Part 1 Difficult <a name="part-1-difficult"></a>

## App 1 And App 2 Came From Different Intake Systems

**Problem**  
The two application files answered the same business question but used different column names.

**Why It Mattered**  
App 1 used `Email Address`, while App 2 used `Username` for borrower email.

**What I Tried**  
I created one standardized `borrower_email` field in both files.

**What I Learned**  
Before modeling, I had to build a reliable borrower identity key.

## Category Fragmentation

**Problem**  
Financing labels, property labels, emojis, and punctuation created duplicate-looking categories.

**Why It Mattered**  
One-hot encoding treated similar business meanings as different model columns.

**What I Tried**  
The baseline model used the original categories first so I could measure the problem.

**What I Learned**  
The first model created a useful baseline, but it was not the cleanest stakeholder model.

## Messy Financial Fields

**Problem**  
Money fields included dollar signs, commas, monthly values, `K`, and `million`.

**Why It Mattered**  
The model needs numeric values, not mixed text formats.

**What I Tried**  
I cleaned money values, applied business rules, and capped extreme liquidity.

**What I Learned**  
Financial cleaning had to happen before imputation and modeling.

## Small And Imbalanced Dataset

**Problem**  
The paid class was much smaller than the dropout class.

**Why It Mattered**  
Overall accuracy looked strong, but the model struggled to identify paid borrowers.

**What I Tried**  
I used `class_weight="balanced"` and reviewed precision, recall, F1, and the confusion matrix.

**What I Learned**  
Accuracy alone was not enough. Paid-class recall mattered because paid borrowers were the smaller class.

___

# 08. Part 1 Results <a name="part-1-results"></a>

![Part 1 class balance](/img/posts/part_1/part_1_class_balance.png)

The raw data was imbalanced. Most applicants dropped out.

![Part 1 confusion matrix](/img/posts/part_1/part_1_confusion_matrix.png)

Part 1 confusion matrix:

```text
[[  0  15]
 [  0 129]]
```

Class-level results:

- Paid precision: **0.00**
- Paid recall: **0.00**
- Paid F1: **0.00**
- Dropout precision: **0.90**
- Dropout recall: **1.00**
- Dropout F1: **0.95**
- Macro F1: **0.47**
- Weighted F1: **0.85**
- Balanced accuracy: **0.50**

![Part 1 feature importance](/img/posts/part_1_feature_importance.png)

The baseline model had **225 columns**, mostly because raw categories were one-hot encoded into many features.

___

# 09. Why Accuracy Was Misleading <a name="accuracy-misleading"></a>

Part 1 testing accuracy was **89.58%**, but that did not mean the model was reliable.

The model predicted every test borrower as dropped out. That made dropout recall high, but paid recall was **0.00**.

The honest interpretation is:

> The model identifies most dropout cases, but its ability to recognize the smaller paid-borrower class remains limited.

___

# 10. 395 Fall_Out_Part_2 <a name="part-2"></a>

Part 2 started from `Book3.xlsx`, the cleaned dataset created after the first workflow.

## State Regions

```python
borrower_data["state_clean"] = borrower_data["state_clean"].replace(state_name_map)
borrower_data = borrower_data[borrower_data["state_clean"] != "Unknown"].copy()
borrower_data["state_region"] = borrower_data["state_clean"].apply(region_from_state)
```

Individual states were grouped into broader U.S. regions. This reduced noise from state-level categories.

## Financing Type Standardization

```python
def clean_financing_type(value):
    value = str(value).lower()

    if "foreclosure" in value:
        return "Refinance"
    if "mid rehab" in value:
        return "Refinance"
    if any(word in value for word in ["refi", "refinance", "cash-out", "cash out"]):
        return "Refinance"
    if "construction" in value:
        return "New Construction"
    if any(word in value for word in ["fix", "flip", "rehab", "renovation"]):
        return "Renovation"
    if any(word in value for word in ["purchase", "bridge", "buy"]):
        return "Purchase"
    return "Unknown"
```

Foreclosure bailout and mid-rehab records were treated as refinance-related categories.

## Property Listing Categories

```python
borrower_data["property_type"] = (
    borrower_data["property_type"].fillna("Unknown").astype(str).str.strip()
)

borrower_data["property_listing_category"] = (
    borrower_data["property_type"].apply(property_listing_bucket)
)
```

Property types were simplified into:

- Single-Family Listing
- Commercial Listing
- Unique Property

## Probability And Risk Buckets

```python
borrower_data["dropout_probability"] = dropout_model.predict_proba(X_all_model)[:, 1]
borrower_data["dropout_probability_percent"] = (
    borrower_data["dropout_probability"] * 100
).round(2)
borrower_data["dropout_risk_bucket"] = borrower_data["dropout_probability"].apply(risk_bucket)
```

This turns the model into a stakeholder-friendly scoring tool.

___

# 11. Part 2 Results <a name="part-2-results"></a>

![Part 2 financing standardization](/img/posts/part_2_financing_before_after_standardization.png)

Financing categories were reduced from **60** raw labels to **5** standardized labels.

![Part 2 property standardization](/img/posts/part_2_property_before_after_standardization.png)

Property categories were reduced from **83** raw labels to **3** stakeholder-friendly groups.

![Part 2 confusion matrix](/img/posts/part_2_confusion_matrix.png)

Part 2 confusion matrix:

```text
[[  0  19]
 [  6 153]]
```

Class-level results:

- Paid precision: **0.00**
- Paid recall: **0.00**
- Paid F1: **0.00**
- Dropout precision: **0.89**
- Dropout recall: **0.96**
- Dropout F1: **0.92**
- Macro F1: **0.46**
- Weighted F1: **0.83**
- Balanced accuracy: **0.48**

![Part 2 feature importance](/img/posts/part_2_feature_importance.png)

The strongest drivers were numeric borrower fields such as credit score, annual income, requested loan amount, liquidity, and real estate experience.

___

# 12. Part 1 vs Part 2 <a name="comparison"></a>

| Area | Part 1 | Part 2 |
|---|---:|---:|
| Source data | Three raw files | Book3.xlsx |
| Purpose | Build target and baseline | Standardize and improve generalization |
| Rows used | 716 | 710 |
| Financing categories | 59 | 5 |
| Property categories | 82 | 3 |
| Geographic feature | Individual state | State region |
| Missing-value method | KNN imputation | Training medians |
| Training accuracy | 1.0000 | 0.9474 |
| Testing accuracy | 0.8958 | 0.8596 |
| Accuracy gap | 0.1042 | 0.0878 |
| Balanced accuracy | 0.5000 | 0.4811 |
| Paid-class recall | 0.0000 | 0.0000 |
| Dropout-class recall | 1.0000 | 0.9623 |
| Macro F1 | 0.4725 | 0.4622 |
| Weighted F1 | 0.8466 | 0.8258 |

![Testing accuracy comparison](/img/posts/comparison_testing_accuracy.png)

Part 2 did not chase a higher accuracy score. Its main improvement was reducing noise and making the model easier to explain.

![Categorical model columns comparison](/img/posts/comparison_categorical_model_columns.png)

The biggest improvement was explainability: Part 2 reduced the number of categorical model columns.

___

# 13. Model Interpretation <a name="interpretation"></a>

Feature importance does not prove causation. It shows which fields the Random Forest used most often to split borrower records.

In Part 2, the top features were mostly numeric:

- `credit_score`
- `annual_income`
- `requested_loan_amount`
- `liquidity_capped`
- `real_estate_experience`

This was a good sign because the improved model was less dominated by individual states or fragmented category labels.

___

# 14. Probability And Risk Buckets <a name="probability-risk"></a>

The model created:

- `predicted_dropped_out`
- `dropout_probability`
- `dropout_probability_percent`
- `dropout_risk_bucket`

Risk bucket rules:

- 80% or higher = High Dropout Risk
- 50% to 79.99% = Medium Dropout Risk
- Below 50% = Low Dropout Risk

![Part 2 risk bucket distribution](/img/posts/part_2_risk_bucket_distribution.png)

Risk buckets are useful for prioritizing outreach, but they should not be used as final business decisions without validation.

___

# 15. Bootstrapping Explanation <a name="bootstrapping"></a>

Part 2 also created a bootstrapped table of 5,000 scored leads.

This must be explained honestly:

- The rows were sampled with replacement.
- Existing borrower patterns may appear more than once.
- The bootstrap table is useful for demonstration, scoring, interface testing, and stakeholder visualization.
- It does not add new real-world information.
- It should not be used to claim stronger validation performance.
- Model testing must remain based on the original held-out data.

![Bootstrap risk distribution](/img/posts/part_2_bootstrap_5000_risk_distribution.png)

___

# 16. Business Impact <a name="business-impact"></a>

The useful business output is not just a class prediction. It is a ranked probability and risk bucket that can support follow-up strategy.

For example:

- High dropout risk borrowers may need faster outreach.
- Medium risk borrowers may need reassurance, documentation help, or a success story.
- Low risk borrowers may need fewer interruptions and a clear call-to-action.

I would not recommend using this model alone for production decisions yet because the paid-borrower class needs stronger validation.

___

# 17. Web Application <a name="web-application"></a>

The local Streamlit app is a proof-of-concept interface for stakeholder scoring. It should load the saved model package, collect borrower inputs, create a single-row DataFrame with the model feature names, and return a dropout or completion probability.

The app should not retrain the model. It should only perform inference.

___

# 18. Limitations <a name="limitations"></a>

This project is a local proof-of-concept, not a production system.

Main limitations:

- The paid class is much smaller than the dropout class.
- Both held-out tests had 0.00 paid-class recall.
- The model may overfit in Part 1 because training accuracy was higher than testing accuracy.
- Bootstrapped rows do not create new borrower evidence.
- Future validation should use new borrower data collected after this modeling period.

___

# 19. Growth/Lessons Learned <a name="lessons-learned"></a>

This project taught me that the hardest part was not the Random Forest model. The hardest part was building a trustworthy target and making messy business data understandable.

I learned to separate:

- **Overfitting:** the model performs much better on training data than unseen testing data.
- **Data leakage:** information connected to the answer enters the training features, so I'll be more cautious of preprocessing data from larger datasets.
- **Category fragmentation:** the same real-world category appears under several labels.
- **Class imbalance:** there are far more dropout rows than paid rows. This gives me the ability to expand this classification method into other areas such as marketing.

___

# 20. Reproducibility <a name="reproducibility"></a>

Run Part 1:

```bash
python3 borrower-fallout-prediction/src/395_Fall_Out_Part_1.py
```

Run Part 2:

```bash
python3 borrower-fallout-prediction/src/395_Fall_Out_Part_2.py
```

Build comparison charts and the data-lineage diagram:

```bash
python3 borrower-fallout-prediction/src/create_case_study_assets.py
```

Project folder:

```text
borrower-fallout-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── CHANGELOG.md
├── src/
│   ├── 395_Fall_Out_Part_1.py
│   ├── 395_Fall_Out_Part_2.py
│   └── create_case_study_assets.py
├── data/
├── outputs/
├── visuals/
├── models/
├── portfolio/
└── backups/
```

Final business summary:

This case study shows a full borrower-fallout modeling workflow from raw application and payment records to model scoring and stakeholder visuals. The final model is useful as a business prototype and portfolio demonstration, but it should not be called production-ready until the paid-borrower class improves and the model is validated on future data.
