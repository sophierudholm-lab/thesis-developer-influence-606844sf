# thesis-developer-influence-606844sf
R codebase and machine learning pipelines for my Master's thesis on predicting B2B SaaS purchase influence.

# Predicting Developer Purchase Influence: A Machine Learning Approach to B2B SaaS Targeting

**Author:** Student ID 606844  
**Degree:** Master Thesis Data Science & Marketing Analytics  
**Institution:** Erasmus University Rotterdam, Erasmus School of Economics  
**Supervisor:** Pieter Schoonees  
**Second Assessor:** Martijn de Jong  

---

## 📌 Overview

This repository contains the R programming code and analytical pipelines for the thesis investigating how to identify developer purchase influence from observable characteristics. 

As B2B software procurement shifts toward Product-Led Growth (PLG), end-users like developers increasingly act as informal purchase influencers. This project utilizes the 2023 Stack Overflow Developer Survey (n=51,551) to develop a machine-learning framework that predicts which developers report high purchase influence. 

The findings offer both descriptive and predictive insights into modern software procurement, equipping B2B marketing and sales teams with a targeted approach to user-based lead scoring.

---

## 📂 Repository Structure

The analysis is streamlined into two primary R scripts:

| File | Description |
| :--- | :--- |
| `main_analysis.R` | Contains the complete pipeline for RQ1 (Predicting Purchase Influence using XGBoost, Random Forest, and Elastic Net) and RQ2 (Developer Persona Clustering using k-prototypes). Generates all primary metrics, SHAP values, lift/gains charts, and cluster profiles. |
| `robustness_check.R` | Replicates the supervised machine learning pipeline (RQ1) using an alternative encoding for the target variable (grouping "some influence" with "great deal of influence") to validate the stability of the model's feature importance. |

---

## ⚙️ System Requirements & Setup

The analysis is conducted entirely in **R** relying heavily on the `tidymodels` ecosystem for predictive modeling and `clustMixType` for mixed-data segmentation. 

**Core Dependencies:**
* `tidyverse`
* `tidymodels`
* `xgboost`
* `ranger`
* `glmnet`
* `shapviz`
* `clustMixType`

---

## 🚀 Usage & Reproducibility

**1. Data Acquisition**
Due to GitHub's file size limits, the raw dataset is not included in this repository. 
* Download the **2023 Stack Overflow Developer Survey** (available under the Open Database License at [https://survey.stackoverflow.co/](https://survey.stackoverflow.co/)).
* Extract the `.csv` file containing the main survey results.
* Rename the file to exactly `results_2023.csv` and place it in the same root directory as the R scripts.

**2. Running the Code**
Run `main_analysis.R` to execute the full data cleaning, feature engineering, and modeling pipeline. The script is configured to use a fixed random seed (`2026`) for 5-fold cross-validation and bootstrap sampling to ensure exact reproducibility.

---

## 📜 Disclaimer

The content of this thesis and repository is the sole responsibility of the author and does not reflect the view of the supervisor, second assessor, Erasmus School of Economics, or Erasmus University.
