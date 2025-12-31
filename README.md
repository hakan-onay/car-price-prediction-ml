# 🚗 Car Price Prediction with Machine Learning

This project predicts used car prices using multiple machine learning models.
It follows a complete end-to-end machine learning pipeline including data
cleaning, preprocessing, model training, evaluation, and visualization.

---

## 📌 Project Overview

- **Goal:** Predict the selling price of used cars
- **Dataset:** CarDekho Used Car Dataset
- **Target Variable:** `selling_price`
- **Models Used:** Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- **Evaluation Metrics:** R², RMSE, MAE, Cross-Validation R²
- **Best Model:** Automatically selected based on Test R²

---

## 📂 Project Structure

```
CarPricePrediction/
│
├── cardekho_dataset.csv
├── main.py
├── requirements.txt
├── plots/
│   ├── correlation_matrix.png
│   ├── random_forest_performance.png
│   ├── overfitting_control_analysis.png
│   └── feature_importance.png
│
├── report/
│   └── Car_Price_Prediction_Report.pdf
│
└── README.md
```

---

## 🔄 Machine Learning Pipeline

### 1. Data Loading

- Dataset loaded from CSV using Pandas
- Initial inspection of shape and columns

### 2. Data Cleaning

- Removed unnecessary columns
- Removed missing values
- Cleaned rows are reported with percentage

### 3. Outlier Detection

- IQR (Interquartile Range) method applied on `selling_price`
- Extreme price values removed

### 4. Feature Engineering

- **Categorical variables:** brand, seller_type, fuel_type, transmission_type
- **Numerical variables:** mileage, engine, seats, max_power, etc.
- **High-cardinality column (`model`)** encoded using Label Encoding
- StandardScaler applied to numerical features
- OneHotEncoder applied to categorical features

### 5. Train–Test Split

- 80% training / 20% test split
- Fixed random state for reproducibility

---

## 🧠 Models Implemented

- Linear Regression
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- Random Forest Regressor
- Gradient Boosting Regressor

All models are implemented using **Scikit-learn Pipelines**.

---

## 📊 Model Evaluation

Each model is evaluated using:

- Train R²
- Test R²
- 5-Fold Cross-Validation R²
- RMSE
- MAE
- Overfitting analysis (Train vs Test difference)

Automatically generated plots:

- Correlation matrix
- Actual vs predicted values
- Error distribution
- Overfitting control graphs
- Feature importance (tree-based models)

All plots are saved under the `plots/` directory.

---

## 🏆 Best Model

The best model is selected based on:

- Highest Test R²
- Low Train–Test R² difference
- Stable cross-validation performance

Tree-based ensemble models achieved the highest performance with low overfitting risk.

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python main.py
```

Make sure `cardekho_dataset.csv` is in the same directory.

---

## 🎯 Key Takeaways

- Ensemble models outperform linear models
- Proper preprocessing significantly improves accuracy
- Cross-validation is essential for reliable evaluation
- The pipeline follows real-world machine learning standards

---

## 👤 Author

**Hakan Onay**  
Software Engineering Student  
Machine Learning & Data Science

## 📜 License

This project is licensed under the MIT License.
