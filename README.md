# Identifying Developer Purchase Champions in B2B SaaS

A machine learning framework to predict which developers influence software purchasing decisions in their organizations.

## Overview

This project analyzes the 2023 Stack Overflow Developer Survey to identify and profile developers who report high influence over their organization's technology purchases. Using supervised and unsupervised machine learning methods, we predict purchase influence and segment developers into distinct profiles.

## Setup

### Requirements

- R version 4.0 or higher
- Stack Overflow 2023 Developer Survey data (`results_2023.csv`)

### Installation

1. Clone this repository
2. Install required R packages:

```r
packages <- c("tidyverse", "tidymodels", "xgboost", "ranger", "glmnet", 
              "shapviz", "probably", "future.apply", "yardstick", 
              "clustMixType", "mclust")
install.packages(packages)
```

3. Download the 2023 Stack Overflow Developer Survey from [Stack Overflow](https://survey.stackoverflow.co/) and save as `results_2023.csv` in your working directory and change path in code.

## How to Run

### Main Analysis

```r
source("fullCodePurchasingChampions.R")
```

This script performs:
- Data filtering and feature engineering
- Supervised prediction with XGBoost, Random Forest, and Elastic Net
- Developer segmentation using k-prototypes clustering
- Performance evaluation and visualization

### Robustness Check

```r
source("RobustnessCheck.R")
```

This script validates results using an alternative target variable encoding.

## Results

- **Best Model**: XGBoost (ROC-AUC = 0.818)
- **Practical Value**: Developers in top 10% of predictions are 3.42× more likely to report high purchase influence
- **Key Finding**: Organization size has the largest overall contribution to model predictions, with smaller organisations (<100) generally associated with higher predicted purchase influence.

## Files

- `fullCodePurchasingChampions.R` - Main analysis script
- `RobustnessCheck.R` - Validation with alternative encoding
- Thesis PDF - Full documentation and results

## License

This code is released under the MIT License. See LICENSE file for details.

## Citation

```bibtex
@mastersthesis{rudholm2026,
  title={Identifying Developer Purchase Champions in B2B SaaS},
  author={Rudholm, Sophie Fourni\'e},
  year={2026},
  school={Erasmus University Rotterdam}
}
```
