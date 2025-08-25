# Data-Science
Linear Regression in Google Colab
📌 Overview

This project demonstrates Linear Regression using Python in Google Colab. It includes data preprocessing, model training, visualization, and evaluation with metrics like MSE, RMSE, and R² score.

🧮 What is Linear Regression?

Linear Regression models the relationship between dependent and independent variables by fitting a straight line:
y=β0​+β1​x+ϵ

Where:

y → Dependent variable (target)

x → Independent variable (feature)

β0 → Intercept

β1 → Coefficient

ε → Error term

🚀 Features

Simple Linear Regression (1 variable)

Multiple Linear Regression (multi-features)

Data preprocessing (handling missing values, normalization)

Visualization with Matplotlib/Seaborn

Model evaluation metrics (MSE, RMSE, R²)

📂 Project Structure

Since Colab is notebook-based, files are minimal:

├── Linear_Regression.ipynb   # Main Google Colab notebook
├── data/                     # (Optional) dataset uploaded to Colab
└── README.md                 # Documentation

⚙️ Setup in Google Colab

Open Google Colab
.

Upload the Linear_Regression.ipynb notebook.

(Optional) Upload dataset (CSV) to the Colab environment:

from google.colab import files
uploaded = files.upload()  # Choose your dataset.csv


Install dependencies (usually pre-installed in Colab):

!pip install numpy pandas matplotlib seaborn scikit-learn

🖥️ Usage

Open and run each cell in Linear_Regression.ipynb.

Load dataset:

import pandas as pd
df = pd.read_csv("your_dataset.csv")


Train the model using scikit-learn:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X = df[['feature_column']]
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


Evaluate performance:

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)


Visualize regression line:

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red", linewidth=2)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Fit")
plt.show()

📊 Example Output

Regression line plotted over data points

Evaluation metrics like:

MSE: 12.45
RMSE: 3.53
R² Score: 0.87

🔮 Future Improvements

Add Polynomial Regression

Implement Regularization (Ridge, Lasso)

Use cross-validation

👉 Do you want me to also give you a ready-made Colab noteboo
