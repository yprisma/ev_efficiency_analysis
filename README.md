# Modeling Electric Vehicle Efficiency and Performance with Machine Learning
**Tools:** Python, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, Statsmodels  
**Methods:** GMM Clustering, K-Means, OLS Regression, Logistic Regression, RFECV, PCA

---

## Overview
This project uses supervised and unsupervised machine learning to analyze how the technical specifications of electric vehicles (EVs) — such as battery capacity, torque, and aerodynamic dimensions — relate to efficiency and driving range. The dataset covers 478 EV models across 22 technical attributes sourced from Kaggle.

The goal: help both consumers and manufacturers understand what actually drives EV performance, using data rather than marketing.

---

## Key Questions
- Can EVs be naturally grouped into performance tiers based purely on technical specs?
- Which features most influence energy efficiency (Wh/km)?
- Can driving range be reliably predicted from measurable specifications?
- Does drivetrain type (AWD/FWD/RWD) meaningfully affect efficiency?

---

## Methods & Results

### 1. Clustering — Identifying Performance Tiers
Applied **Gaussian Mixture Model (GMM)** and **K-Means** clustering to group EVs by technical specs without using brand or price labels.

- GMM (AIC/BIC evaluation) surfaced 12–20 optimal clusters depending on complexity preference
- K-Means elbow method identified **3 natural performance tiers**: economy, mid-range, and premium
- High-performance brands (Tesla, Lucid, Porsche) clustered separately from mass-market models — driven purely by battery capacity and power output, not brand labels
- Notably, Polestar and Volvo (sister companies) landed in different clusters, reflecting Polestar's EV-native design vs. Volvo's traditional background

### 2. Regression — What Drives Efficiency?
Used **RFECV** to select the 12 most predictive features, then built an **OLS regression** model with `efficiency_wh_per_km` as the target.

- RFECV cross-validation R² = **0.60**
- Full OLS model R² = **0.83** — the selected features explain 83% of efficiency variance
- Key drivers: battery capacity, range, vehicle length, height, towing capacity, and top speed
- **Drivetrain type had no statistically significant effect** (p ≈ 0.83) — a counterintuitive finding that challenges conventional assumptions about AWD/RWD efficiency trade-offs

### 3. Drivetrain Classification
Trained a **logistic regression** classifier to predict drivetrain type (AWD/FWD/RWD) from technical specs alone, across 1,000 random train/test splits.

- Average training accuracy: **~94%**
- Average test accuracy: **~90%**
- Most predictive features: torque, acceleration, and fast-charging capacity

### 4. Range Prediction
Built a **multiple linear regression** model to predict driving range (km) from 11 technical features.

- Cross-validation score: **0.93**
- R² = **0.96** — range is highly predictable from specs like battery capacity, number of cells, and torque
- Top speed, vehicle width, and fast-charging power were not statistically significant predictors

---

## Key Takeaway
EV efficiency and range are driven primarily by **battery characteristics and vehicle size/weight**, not drivetrain configuration. This has practical implications: manufacturers optimizing for efficiency should focus on energy storage and aerodynamic design rather than powertrain layout.

---

## Data Source
[Electric Vehicle Specs Dataset]([https://www.kaggle.com/](https://www.kaggle.com/datasets/urvishahir/electric-vehicle-specifications-dataset-2025)) — 478 EV models, 22 technical attributes

---

## Team
Janie Aguilera, Yishuan Chung, Amanda Kim, & Xanthia Victuelles  
IMT 574 — Data Science II: Machine Learning  
University of Washington Information School, 2025
