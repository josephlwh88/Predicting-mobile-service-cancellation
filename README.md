# Telecom Churn Prediction & Analytics

An end-to-end machine learning pipeline that identifies high-risk churn subscribers using behavioral metrics, usage patterns, and support interaction logs. Built with Python, `scikit-learn`, `pandas`, and `seaborn`.

---

## Executive Summary

Customer churn is a critical revenue risk in telecommunications. This project develops a Random Forest classification pipeline that analyzes subscriber usage and customer service touchpoints to identify churn drivers and predict at-risk accounts.

* **Test Accuracy:** 98.20%
* **ROC-AUC Score:** 0.919
* **Precision:** 100% (Zero false positives on held-out test data)
* **Recall:** 87.37% (Captures 83 out of 95 churners in the test set)

---

## Business Insights & Key Drivers

* **Customer Service Tipping Point:** Subscribers making **$\ge 4$ customer service calls** exhibit a sharp increase in churn probability (jumping from ~10.6% to over 48%), making support contact frequency the strongest operational indicator of friction.
* **Billing Volume & Day Usage:** Total account expenditure (`Total charge`) and high peak-time minute usage (`Total day charge`) drive nearly **30%** of total model importance. High-spending accounts are significantly more sensitive to pricing and competitor offers.
* **Plan Friction:** Customers enrolled in an **International Plan** churn at almost four times the rate of standard subscribers, highlighting potential expectations or rate-structure issues.

---

Feature Selection & Engineering StrategyTo maximize model signal and prevent overfitting, feature engineering and explicit feature selection were applied:Engineered FeaturesTotal charge: Sum of day, evening, night, and international charges.Total minutes: Sum of day, evening, night, and international call durations.Total calls: Sum of all call attempt counts across time windows.CS_Calls_HighRisk: Binary threshold flag ($1$ if Customer service calls $\ge 4$, else $0$).Selected Features (12)Uninformative spatial features (State, Area code) and low-variance counters (Account length, individual call counts) were dropped.Total charge, International plan, Customer service calls, CS_Calls_HighRisk, Total day minutes, Total day charge, Total minutes, Total intl calls, Total intl minutes, Total intl charge, Number vmail messages, Voice mail plan.


## Project Structure

```text
├── churn-bigml-80.csv           # Training dataset (80% / 2,666 records)
├── churn-bigml-20.csv           # Held-out testing dataset (20% / 667 records)
├── main.py                      # Complete execution pipeline script
├── README.md                    # Project documentation
└── assets/                      # Generated visualization outputs
    ├── rf_classifier_outputs.png
    └── rf_single_tree.png
