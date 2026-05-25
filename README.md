# Customer Churn Prediction & Data Pipelines: Data Mining

**Authors:** Bezalel Spolter & Eden Filosof

**Overview:** This repository contains a comprehensive two-part Data Mining and Machine Learning project focused on analyzing e-commerce customer behavior to predict customer churn. Part 1 covers exhaustive exploratory data analysis (EDA), anomaly discovery, and statistical data cleansing pipelines. Part 2 develops, scales, and evaluates predictive machine learning classification models (Random Forest, AdaBoost, Support Vector Machines, and XGBoost) along with a terminal ensemble Voting Classifier.

---

## Part 1: Exploratory Data Analysis & Advanced Data Preprocessing

### 1. Initial Profile & Feature Categorization

We analyzed feature data types, missing value distributions, and geometric spreads across all columns. The dataset was partitioned into three explicit groups:

* **Numerical Features:** Continuous variables optimized for parametric statistical distribution profiling.
* `age`, `avg_frequency_login_days`, `avg_time_spent`, `avg_transaction_value`, `days_since_last_login`, `points_in_wallet`.


* **Categorical Features:** Qualitative variables mapped to discrete groups.
* `churn` (Target), `complaint_status`, `feedback`, `gender`, `internet_option`, `joined_through_referral`, `joining_date`, `medium_of_operation`, `membership_category`, `past_complaint`, `preferred_offer_types`, `offer_application_preference`, `region_category`, `used_special_discount`.


* **Identification / High Cardinality Columns:** Structural variables with completely unique values across rows.
* `referral_id`, `customer_id`, `Name`, `security_no`, `Unnamed: 0`.



### 2. Anomaly Cleansing & Outlier Truncation

To clean the data before individual feature manipulation, row-level anomaly scores were computed by summing the count of invalid attributes per record (e.g., missing fields, string "Error" values, negative values in bounded metrics).

A frequency check revealed a distinct cluster of highly broken rows: **419 records contained 3 or 4 invalid properties simultaneously.** These records were dropped to protect the integrity of downstream distributions. Additionally, strict outlier cleaning removed all rows containing values straying beyond $\pm3$ standard deviations ($3\sigma$) across any numerical column.

### 3. Cross-Feature Synthesis & Target Variable Insights

Cross-referencing customer data against the `churn` target revealed critical behavioral patterns:

* **Feedback Correlation:** Customers who churned (`1`) left exclusively negative or unknown feedback. Zero churned users left positive feedback. Retained users (`0`) span all feedback categories.
* **Loyalty Tier Boundaries:** Churned users held either `Basic Membership` or no membership at all. Premium loyalty tiers showed near-zero churn.
* **Purchase Power Limits:** Retained users frequently crossed the 50,000 purchase value mark, whereas churned users rarely did.
* **Loyalty Points Deficits:** Churned users averaged fewer wallet points (~600) compared to retained users (~800).
* **Offer Engagement Drivers:** Cross-analyzing `membership_category` and `preferred_offer_types` revealed that non-members and basic members actively seek marketing offers due to high price sensitivity. Conversely, Premium and Platinum members show little engagement with generic offers since their tiers already include native perks.

---

## Part 2: Machine Learning Classification & Model Optimization

### 1. Preprocessing & Feature Engineering Pipeline

Before model training, the cleaned dataset (`data_df`) underwent automated feature engineering and matrix transformations:

1. **Feature Derivation (`days_since_joined`):** Extracted from `joining_date` by calculating the exact delta between the customer's acquisition date and a static timeline baseline (January 1, 2018).
2. **High-Cardinality Removal:** Dropped descriptive identifying attributes (`Name`, `customer_id`, `joining_date`) to prevent overfitting.
3. **Numerical Scale Standardization:** Applied `MinMaxScaler` transformation to scale all continuous numerical features precisely between `0` and `1`.
4. **Categorical Vectorization:** Converted all qualitative columns into binary vectors via One-Hot Encoding (`True`/`False`).
5. **Test Set Pipeline Alignment:** The raw test dataset (`test`) was processed through the exact same transformations to ensure zero data leakage.
6. **Stratified Data Split:** Partitioned `data_df` into train (80%: `X_train`, `y_train`) and validation sets (20%: `X_test`, `y_test`), preserving the 75/25 target class ratio across both splits.

### 2. Optimization Strategy & Evaluation Framework

Models were evaluated using three primary metrics: **Accuracy** (overall reliability), **Precision** for Churn=1 (protecting marketing budgets by minimizing false alarms), and **Recall** for Churn=1 (ensuring actual churners are caught early for retention interventions).

The optimization workflow used 5-Fold Cross-Validation ($k=5$). Hyperparameters were tuned using Grid Search or Random Search (running 50 iterations to manage compute time). Training vs. validation curves for loss and accuracy confirmed zero overfitting and strong generalization.

### 3. Machine Learning Algorithms & Hyperparameter Spaces

* **Random Forest:** Tuned using Random Search over `n_estimators` ($50 \rightarrow 400$), `max_depth` ($10 \rightarrow 30$), `bootstrap` (`True`, `False`), splitting `criterion` (`gini`, `entropy`), and `class_weight` (`None`, `balanced`).
* **AdaBoost:** Tuned over `n_estimators` ($50 \rightarrow 400$) and `learning_rate` ($0.001 \rightarrow 0.1$).
* **Support Vector Machines (SVM):** Optimized over alternative `kernel` selections: Polynomial (evaluating degrees 3 and 5), Radial Basis Function (RBF Gaussian), and Sigmoid.
* **XGBoost:** Implemented parallel tree-based gradient boosting with explicit regularization optimization over `n_estimators` ($50 \rightarrow 400$), `max_depth` ($5 \rightarrow 11$), `learning_rate` ($0.001 \rightarrow 0.1$), $L1$ regularization (`reg_alpha`: $0 \rightarrow 0.05$), and $L2$ regularization (`reg_lambda`: $1 \rightarrow 2$).

### 4. Comprehensive Operational Results

#### Scenario A: Maximizing Overall Accuracy (Model Reliability)

This configuration targets the highest overall predictive precision across both classes.

| Classifier Model | Optimal Hyperparameter Configuration | Accuracy | Secondary Metric Performance Shifts | Top Extracted Feature Importances |
| --- | --- | --- | --- | --- |
| **Random Forest** | `n_estimators=400`, `max_depth=10`, `criterion='entropy'`, `class_weight='balanced'`, `bootstrap=True` | **86.6%** | High Recall (Churn=1) & Precision (Churn=0) at **99%**. However, Precision for Churn=1 is low (**68%**), leading to false positives. | 1. `points_in_wallet`<br>

<br>2. `membership_category` (Gold/Silver) |
| **AdaBoost** | `n_estimators=400`, `learning_rate=0.1` | **85.5%** | Highly balanced performance for Churn=1: both Precision and Recall stabilize at **~72%**. | *N/A* |
| **SVM** | `kernel='poly'`, `degree=3` | **84.7%** | Test set Precision for Churn=0 reaches **100%**. Precision for Churn=1 drops to **65%**. | *N/A* |
| **XGBoost** | `learning_rate=0.05`, `max_depth=7`, `n_estimators=100`, `reg_alpha=0.01`, `reg_lambda=1` | **86.5%** | Highly balanced metrics across all customer behavior splits. | 1. `membership_category` (Gold/Silver)<br>

<br>2. `points_in_wallet` |
| **Voting Ensemble** | Majority Voting blend combining the top 3 tuned classifiers above. | **86.9%** | Secures the highest overall validation accuracy of the entire project. | *N/A* |

#### Scenario B: Maximizing Precision for Churn=1 (Targeted Marketing Efficiency)

This configuration prioritizes high-confidence predictions to minimize false positives and avoid wasting retention budgets on loyal customers.

* **Random Forest:** `n_estimators=300`, `max_depth=None`, `criterion='gini'`, `class_weight='balanced'`, `bootstrap=False`
* *Results:* **Precision (Churn=1): 72%** | Accuracy: 86% | Recall (Churn=1): 77% (misses 23% of true churners).


* **SVM:** `kernel='poly'`, `degree=3`
* *Results:* **Precision (Churn=1): 64%** | Accuracy: 85% | Precision (Churn=0): 100%.


* **XGBoost:** `learning_rate=0.001`, `max_depth=7`, `n_estimators=500`, `reg_alpha=0`, `reg_lambda=2`
* *Results:* **Precision (Churn=1): 97%** | Accuracy: 83% | Recall (Churn=1): 37% (conservative model that misses 63% of true churners).



#### Scenario C: Maximizing Recall for Churn=1 (Total Churn Capture)

This configuration ensures almost all potentially churning users are caught, accepting higher false alarms as a trade-off for safety.

* **Random Forest:** `n_estimators=200`, `max_depth=10`, `criterion='entropy'`, `class_weight='balanced'`, `bootstrap=False`
* *Results:* **Recall (Churn=1): 99%** | Metrics align with the accuracy-maximized Random Forest setup.


* **SVM:** `kernel='poly'`, `degree=3`
* *Results:* **Recall (Churn=1): 100%** | Precision (Churn=1): 65% | Accuracy: 85%. Captures the entire churn cohort perfectly.


* **XGBoost:** `learning_rate=0.1`, `max_depth=5`, `n_estimators=30`, `reg_alpha=0.05`, `reg_lambda=1`
* *Results:* **Recall (Churn=1): 95%** | Accuracy: 87% | Precision (Churn=1) drops to 68%.



---
