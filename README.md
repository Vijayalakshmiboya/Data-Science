🚗 Car Price Prediction (Elastic Net Regression)
📌 Overview

This project predicts **car prices** using **Elastic Net Regression**, which combines **L1 (Lasso)** and **L2 (Ridge)** regularization. The model achieved an **R² Score of \~0.79** on the test set.

## ⚙️ Tech Stack

* **Python** 🐍
* **pandas, scikit-learn, matplotlib**


## 📂 Dataset

`Car_Price_Prediction.csv` includes:

* **Price** (Target)
* Car details: *Make, Model, Fuel Type, Transmission, etc.*



## ▶️ Quick Start

```bash
# Install dependencies
pip install pandas scikit-learn matplotlib
```

Run the notebook in **Google Colab** and upload your dataset.


## 📊 Results

* **Mean Squared Error (MSE):** `5,737,909.27`
* **R² Score:** `0.79`

📈 Visualization: *Actual vs Predicted Car Prices*


## 🚀 Future Work

* Hyperparameter tuning (`GridSearchCV`)
* Compare with **Linear, Ridge, Lasso, Random Forest**
* Add more car features (*year, mileage, engine size*)


