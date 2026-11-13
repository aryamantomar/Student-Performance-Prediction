# Student Academic Performance Prediction Using Machine Learning

**Author:** Aryaman Tomar  

An end-to-end, beginner-friendly yet academically rigorous Machine Learning project predicting secondary school students' final mathematics examination grades (`G3`) using demographic, social, and academic attributes.

---

## 📁 Project Deliverables

| File | Description |
| :--- | :--- |
| **[`Student_Performance_Prediction.ipynb`](file:///home/aryaman-tomar/student_performance_prediction/Student_Performance_Prediction.ipynb)** | Complete, fully executed Jupyter Notebook containing all 10 required sections, formatted tables, visualizations, and an interactive prediction tool. |
| **[`student-mat.csv`](file:///home/aryaman-tomar/student_performance_prediction/student-mat.csv)** | The primary UCI Mathematics Student Performance dataset (395 records, 33 attributes). |
| **[`README.md`](file:///home/aryaman-tomar/student_performance_prediction/README.md)** | Step-by-step instructions on running the notebook in Google Colab and local environments. |

---

## 🚀 How to Run the Notebook

### Option A: Running in Google Colab (Recommended for zero setup)

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Click **File** > **Upload notebook** and upload [`Student_Performance_Prediction.ipynb`](file:///home/aryaman-tomar/student_performance_prediction/Student_Performance_Prediction.ipynb).
3. Click **Runtime** > **Run all** (or press `Ctrl + F9`).
4. **Zero Manual Setup Required:** Section 3 includes an automated download routine that fetches the dataset directly from the UCI archive or mirror if `student-mat.csv` is not present in the runtime folder.

### Option B: Running Locally with Jupyter Notebook / VS Code

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/aryaman-tomar/student_performance_prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook Student_Performance_Prediction.ipynb
   ```
   Or open the folder directly in VS Code / Cursor / PyCharm with the Jupyter extension installed.

5. Select the `.venv` kernel and click **Run All**.

---

## 🔬 Project Architecture & Workflow

```
1. Ingestion & Validation
   └── Auto-downloads UCI Student Performance data (student-mat.csv)
       └── 395 samples, 33 columns, 0 missing values, 0 duplicates
2. Exploratory Data Analysis (EDA)
   ├── Target Grade Distribution (G3: Mean=10.42, 9.62% scoring 0 due to dropouts)
   ├── Study Time Impact (Upward trend: 8.85 for <2 hrs vs 11.26 for >10 hrs)
   ├── School Absences Correlation (Subtle r=+0.034 due to dropout reporting differences)
   ├── Academic Continuity (G1: r=0.801, G2: r=0.905 strong positive correlations)
   └── Numerical Correlation Matrix (Failures: r=-0.36; Alcohol consumption: r=-0.14)
3. Data Preprocessing & Pipeline Construction
   ├── 80:20 Train-Test split (random_state=42)
   ├── Strict data leakage prevention (Transformers fitted strictly on X_train)
   ├── ColumnTransformer:
   │   ├── StandardScaler for numerical predictors (15 features)
   │   └── OneHotEncoder(drop='first') for categorical features (17 features -> 26 encoded)
4. Model Training & Comparison
   ├── 1. Linear Regression (Parametric baseline)
   ├── 2. Decision Tree Regressor (max_depth=5)
   └── 3. Random Forest Regressor (100 estimators, ensemble bagging)
5. Model Evaluation (Actual Metrics on Test Set)
   ├── Mean Absolute Error (MAE)
   ├── Mean Squared Error (MSE)
   ├── Root Mean Squared Error (RMSE)
   └── R² Score
6. Diagnostics & Feature Importance
   ├── Champion Model: Random Forest (R² = 0.813, MAE = 1.179, RMSE = 1.958)
   ├── Actual vs. Predicted scatter plot with 45° reference line
   ├── Residual analysis (Residuals vs Predicted & Error Distribution)
   └── Top 15 Feature Importances (G2: ~79.3%, Absences: ~11.0%, G1: ~0.6%)
7. Interactive Inference System
   └── predict_student_grade() function with [0, 20] boundary clipping and academic tiering
```

---

## 📊 Summary of Actual Model Evaluation Results

All metrics were computed strictly on the held-out test set ($20\\%$, $79$ students):

| Rank | Model | MAE (Grade Points) | MSE | RMSE (Grade Points) | Test $R^2$ Score | Train $R^2$ Score | Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Random Forest Regressor** | **1.179** | **3.835** | **1.958** | **0.8130** | 0.9701 | **Champion:** Lowest error, captures non-linear synergies, highest explained variance. |
| 🥈 | **Linear Regression** | 1.598 | 5.528 | 2.351 | 0.7304 | 0.8447 | Strong baseline, captures linear momentum from G1 and G2. |
| 🥉 | **Decision Tree Regressor** | 1.357 | 6.087 | 2.467 | 0.7031 | 0.9026 | Higher variance and step discretization; sensitive to individual splits. |

---

## 💡 Plain-Language Metric Guide

- **MAE (Mean Absolute Error):** The average error in actual grade points. A Random Forest MAE of $1.179$ means predictions are typically within $\approx 1.2$ points on a $0$–$20$ grade scale.
- **MSE (Mean Squared Error):** Averages squared differences; penalizes catastrophic outliers heavily.
- **RMSE (Root Mean Squared Error):** The standard deviation of the residuals. Expressed in original grade units ($1.958$ points).
- **$R^2$ (Coefficient of Determination):** Explains that **$81.3\\%$** of the variance in final student grades is accounted for by the student's attributes and historical academic momentum.

---

## 🔮 Interactive Prediction Example

The notebook features a standalone prediction utility:

```python
# Predict for any student profile:
result = predict_student_grade(champion_pipeline, custom_student)
print(f"Predicted Grade: {result['Predicted_G3']} / 20.00")
print(f"Academic Tier:   {result['Academic_Tier']}")
print(f"Guidance:        {result['Recommendation']}")
```

**Output:**
```
Predicted Grade: 14.09 / 20.00
Academic Tier:   Good (Commendable)
Guidance:        Strong performance; encourage consistent effort in problem-solving areas.
```
