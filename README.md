# ML-Project-Compilence-Radar

# Compliance Radar – Machine Learning Project

## 1. Introduction

The Compliance Radar project focuses on analyzing organizational compliance behavior using a rich dataset of operational, risk-related, and audit-related variables.  
The goal is to build an analytical foundation that will allow the team to:

- understand compliance patterns,
- prepare the data for machine learning,
- explore structural relationships between variables,
- and later build models that estimate or classify compliance risk.

This repository contains the initial stages of the project, including data loading, cleaning, preprocessing, and exploratory analysis.

---

## 2. Methods (Current Progress)

### 2.1 Data Source

The data is stored in a SQLite database (`org_compliance_data.db`), which contains four tables:

- **departments** — the primary dataset used for analysis.
- **risk_summary_by_division** — supplementary aggregated metrics.
- **high_risk_departments** — external list of flagged departments.
- **data_dictionary** — definitions and descriptions of all variables.

The **departments** table (709 rows) is the main focus of the analysis.

---

### 2.2 Data Loading

The SQLite database was loaded using Python’s `sqlite3` module.  
All tables were inspected, and the `departments` table was selected as the main analytical dataset.

Additional information about feature meanings was taken from the `data_dictionary` table.

---

### 2.3 Data Cleaning

#### **Handling Missing Values**

The dataset contained substantial missingness (up to ~40% in several variables).  
Since dropping rows would result in severe data loss, the following imputation strategy was applied:

- **Median imputation** for numerical columns  
- **Most frequent category (mode)** imputation for categorical columns  
- Identifier fields (`dept_id`, `dept_name`) were left untouched

After imputation, the dataset contained **no missing values**.

---

### 2.4 Outlier Analysis

Outlier detection was performed using the **IQR (Interquartile Range)** method.  
Many variables showed large numbers of outliers due to the nature of compliance and operational behavior (risk exposure, reporting delays, violations, audit results, etc.).

These values represent real departmental behavior, not errors.  
**Therefore, no outlier removal or clipping was applied.**

---

### 2.5 Encoding Categorical Variables

Categorical features such as:

- department type  
- division  
- location type  
- team size  
- reporting structure  
- functional classifications  

were encoded using **one-hot encoding**.  
Identifier fields (`dept_id`, `dept_name`) were excluded from encoding.

The resulting dataset is entirely numeric (except for the two ID fields).

---

### 2.6 Scaling

All numerical variables were standardized using a manual z-score transformation:

\[
z = \frac{x - \text{mean}}{\text{std}}
\]

Scaling was applied to:

- operational metrics  
- risk indicators  
- audit scores  
- reporting metrics  
- binary flags  
- one-hot encoded features  

`dept_id` and `dept_name` were excluded.

This prepares the dataset for models that are sensitive to feature magnitude, such as clustering or logistic regression.

---

## 3. Exploratory Data Analysis (Current Progress)

Initial EDA has been performed, including:

### **3.1 Correlation Analysis**
A correlation matrix and heatmap were generated to evaluate linear relationships among numerical variables.

### **Distribution Analysis**
Histograms were created for all numerical columns to inspect:

- skewness  
- spread  
- multimodality  
- extreme values  

### **Boxplots**
Boxplots were generated for:

- Risk indicators  
- Audit performance metrics  
- Reporting delay and gap metrics  

This helps visualize variability and highlight departments with extreme behavior.

---

## 4. The 3 ML Problems

Based on the structure of the dataset and the results of the exploratory data analysis, the Compliance Radar project involves three complementary ML tasks:

1. Unsupervised Clustering

We use clustering (e.g., K-Means) to identify natural groupings of departments based on operational, audit, and risk-related features.
This helps reveal behavioral patterns and high-risk clusters without predefined labels.

2. Risk Score Prediction (Regression)

The **overall_risk_score** serves as a continuous target variable.
We apply regression models to estimate risk levels from operational metrics, allowing us to quantify the drivers of compliance risk.

3. Feature Importance & Risk Drivers

Tree-based models and correlation analysis are used to determine which factors most influence compliance behavior.
This supports interpretability and contributes directly to the final recommendations.

---

## 5. Modeling & Experiments (Person 2)

In this section, we use the cleaned and scaled `scaled` dataframe from Person 1
to train and evaluate machine learning models that classify high-risk departments.

---

### 5.1 Creating the Target Variable

We start from the cleaned and scaled dataframe `scaled` produced in Person 1’s section.
Using the `high_risk_departments` table in the database, we create a binary label:

- 1 = high-risk department  
- 0 = not high-risk

---

### 5.2 Feature Selection and Train/Test Split

From the scaled dataset we drop:
- `dept_id`, `dept_name` (identifiers)
- `overall_risk_score`, `compliance_score_final` (final outcomes that may leak information)
- the target `is_high_risk` from the feature matrix

Then we split the data into train and test sets.

---

### 5.3 Models and Hyperparameters

We train three different models on the scaled features:

1. Logistic Regression  
2. Random Forest  
3. HistGradientBoosting  

For each model we define a small hyperparameter grid and use GridSearchCV with 3-fold cross-validation
to find reasonable settings using F1-score as the main metric.

---

### 5.4 Running GridSearchCV

We run GridSearchCV for each model on the training set and keep the best estimator according to F1-score.

---

### 5.5 Model Evaluation

We evaluate each tuned model on the held-out test set using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion matrix and ROC curve

---

### 5.6 Test Results and Model Comparison

We now evaluate all three models on the test set and summarize the metrics in a comparison table.

---

### 5.7 Feature Importance (Random Forest)

Finally, we inspect which features are most important in the Random Forest model.
Since the input data is already scaled and fully numeric, we can directly use
`feature_importances_` together with the original column names.

---

## 6. Interpretation of Results

The model evaluations performed in the notebook provide insights into how each supervised learning approach behaves when classifying high-risk departments.

---

### 6.1 Overall Model Performance

Three models were trained and tuned through GridSearchCV:

- **Logistic Regression**
- **Random Forest**
- **HistGradientBoosting**

All models achieved reasonably strong performance. The key takeaway is:

- **HistGradientBoosting achieved the best overall test-set results**, with:
  - Accuracy ≈ 0.89  
  - F1-score ≈ 0.83  
  - ROC-AUC ≈ 0.93  

This indicates that it provides the strongest ability to discriminate between high-risk and non-high-risk departments.

The Random Forest also performed well (F1 ≈ 0.81, ROC-AUC ≈ 0.90), while Logistic Regression served as a strong linear baseline (F1 ≈ 0.73, ROC-AUC ≈ 0.88).

---

### 6.2 Class Balance and Error Analysis

The target variable `is_high_risk` is moderately imbalanced (≈ 70% class 0, 30% class 1).

The models show:

- **High recall for high-risk cases**, meaning the model successfully catches most departments that truly belong in the high-risk category.
- Slightly lower precision, meaning some departments are incorrectly flagged as high-risk.

In compliance contexts, this trade-off is acceptable:

- **False negatives (missed high-risk cases)** are more dangerous than  
- **False positives (extra departments being reviewed)**.

---

### 6.3 ROC Curves and Confusion Matrices

The ROC curves show strong separability between the classes, especially for HistGradientBoosting.  
Confusion matrices confirm that:

- Misclassification rates are low.
- The majority of high-risk cases are correctly identified.
- Non-high-risk cases are rarely misclassified.

---

### 6.4 Feature Importance and Risk Drivers

Random Forest feature importance analysis shows that the strongest predictors of risk typically include:

- **Historical violations indicators**
- **Audit results** (`audit_score_q1`, `audit_score_q2`, `compliance_score_final`)
- **Risk exposure measures**
- **Reporting gaps and delays**
- **Remediation and oversight characteristics**

These align with intuitive domain expectations and reinforce the interpretability of the model.

--- 

## 7. Ethical and Organizational Considerations

While the models provide strong predictive performance, deploying a risk classification system inside an organization requires careful attention to ethics, governance, and transparency.

---

### 7.1 Fairness and Potential Bias

Historical compliance data may carry embedded biases — for example, some divisions may have been audited more frequently than others.  
If not monitored, a model may unintentionally reinforce these patterns.

Mitigations include:

- Performance tracking across subgroups (division, department type, location)
- Monitoring false positive and false negative rates by group
- Adjusting thresholds or incorporating fairness constraints when needed

---

### 7.2 Transparency and Explainability

Compliance decisions have regulatory consequences.  
Therefore, the model must remain **explainable**, not a “black box.”

Recommended practices:

- Document all preprocessing steps (imputation, encoding, scaling)
- Use feature importance and modern interpretability tools (e.g., SHAP values)
- Provide users with clear explanations of predictions

---

### 7.3 Human-in-the-Loop Design

The system should support, not replace, compliance experts.

- Predictions should trigger **human review**, not automatic penalties
- Experts must be able to override model outputs
- Feedback from investigators should be logged and incorporated into retraining cycles

---

### 7.4 Data Governance

Because the underlying data contains sensitive operational and audit information:

- Access should be restricted and monitored
- Retention policies should be established
- Model predictions should be logged to ensure accountability

Together, these considerations ensure the Compliance Radar remains a responsible and trustworthy tool.

---

## 8. Recommendations and Next Steps

---

### 8.1 Operational Use Cases

The current prototype already supports several high-value use cases:

- **Risk-based audit prioritization**  
  Identify the departments most likely to require intervention.

- **Continuous monitoring**  
  Recompute risk levels quarterly or monthly to track improvement or deterioration.

- **Management insights**  
  Use risk-driver analysis to identify structural weaknesses across the organization.

---

### 8.2 Threshold Strategy

Depending on business needs, the decision threshold can be tuned to emphasize:

- **Higher recall** when missing high-risk cases is unacceptable  
- **Higher precision** when investigation resources are limited

This threshold process should be documented as part of internal policy.

---

### 8.3 Model Monitoring and Retraining

Once deployed, models must be monitored for:

- **Performance drift** (drop in accuracy, recall, F1)
- **Concept drift** (feature importance shifts)
- **Data drift** (changes in distribution over time)

Regular retraining cycles should be scheduled based on these indicators.

---

### 8.4 Future Enhancements

Recommended improvements include:

- Adding **time-series features** (quarterly trends in audits, violations, etc.)
- Implementing **SHAP-based explainability dashboards**
- Testing alternative models such as XGBoost or LightGBM
- Integrating additional contextual datasets (HR, vendor risk, financial stress indicators)
- Performing **unsupervised clustering** to detect hidden behavioral patterns

Implementing these steps will transform the Compliance Radar into a robust, scalable, and interpretable decision-support system.

