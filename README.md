# Airline Market Fare Analysis

A Python data analysis project exploring 1.58M airline fare records 
from the U.S. Department of Transportation.

---

## Project Overview

This project analyzes airline pricing patterns, detects fare anomalies, 
and builds a predictive model to identify key pricing drivers.

---

## Key Results

- Detected 41,976 pricing anomalies (2.7%) using IQR method
- Trained a Random Forest model achieving R² of 0.9975 and RMSE of $3.67
- Identified route distance and market concentration as top pricing drivers

---

## Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core language |
| pandas | Data cleaning |
| matplotlib / seaborn | Visualizations |
| scikit-learn | Random Forest model |

---

## Data Source

U.S. Department of Transportation — Bureau of Transportation Statistics (BTS)
Dataset: Airline Market Fare Prediction Data (1,581,278 records, 26 columns)

---

## How to Run

pip install pandas matplotlib seaborn scikit-learn

python analysis.py

---

## License

For educational and portfolio purposes only.
