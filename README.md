# 🎓 Student Academic Performance Prediction Using Machine Learning

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aryamantomar/Student-Performance-Prediction/blob/main/Student_Performance_Prediction.ipynb)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Student--Performance--Prediction-black?logo=github)](https://github.com/aryamantomar/Student-Performance-Prediction)

**Author:** [Aryaman Tomar](https://github.com/aryamantomar)  
*Educational Data Mining & Applied Machine Learning*

</div>

---

## 📌 Executive Summary

Academic performance serves as a primary indicator of student scholastic success and future socioeconomic progression. However, students experiencing academic distress often remain undetected until final examinations, when remedial interventions arrive too late.

This project develops an end-to-end, supervised machine learning system that predicts students' final mathematics examination grades (`G3`, continuous scale: 0–20) based on demographic, familial, socio-behavioral, and historical academic attributes. Built on the benchmark **UCI Student Performance Dataset**, this project emphasizes **leakage-free preprocessing pipelines**, empirical benchmark comparisons across multiple regression algorithms, detailed error diagnostics, and an interactive inference engine with automated educational recommendations.

---

## 🗂️ Repository Structure

```plaintext
Student-Performance-Prediction/
├── Student_Performance_Prediction.ipynb   # Main Jupyter Notebook (fully executed with outputs)
├── student-mat.csv                       # Mathematics student dataset (UCI Repository)
├── README.md                             # Academic project documentation & usage guide
└── requirements.txt                      # Python dependencies for local execution
```

---

## 📊 Dataset Overview

The dataset comes from the **UCI Machine Learning Repository**:
- **Dataset:** Student Performance Dataset (Mathematics Course)
- **Primary Source:** Paulo Cortez and Alice Silva, University of Minho, Portugal (2008)
- **Observations:** $395$ secondary school students
- **Features:** $33$ attributes (17 categorical, 15 numerical predictors, 1 target variable)
- **Target Variable:** `G3` — Final grade (integer range: $0$ to $20$)

### Key Feature Categories
| Domain | Key Attributes | Description |
| :--- | :--- | :--- |
| **Academic Momentum** | `G1`, `G2` | Period 1 and Period 2 marks ($0$–$20$) |
| **Academic History** | `failures`, `studytime`, `absences` | Past course failures, weekly study hours, total school absences |
| **Family Background** | `Medu`, `Fedu`, `Mjob`, `Fjob`, `famsup` | Parents' education levels, professions, and family support |
| **Social & Lifestyle**| `Dalc`, `Walc`, `goout`, `freetime`, `health` | Workday/weekend alcohol intake, outings, free time, health rating |
| **Institutional** | `school`, `schoolsup`, `paid`, `activities` | School attended (`GP` / `MS`), extra school support, paid classes |

---

## ⚙️ Methodology & Machine Learning Pipeline

```
Raw Dataset (395 records, 33 attributes)
   │
   ├── 1. Automated Ingestion & Quality Validation (0 missing values, 0 duplicates)
   │
   ├── 2. Exploratory Data Analysis (EDA)
   │     ├── Distribution analysis (Uncovering the 9.62% zero-grade dropout cluster)
   │     ├── Bivariate studies (Study hours, class absences, prior failures)
   │     └── Feature correlation matrix
   │
   ├── 3. Strict Preprocessing (Fitted SOLELY on X_train to prevent Data Leakage)
   │     ├── Train-Test Split (80:20 Stratified Partition, random_state=42)
   │     └── ColumnTransformer:
   │           ├── Numerical Features (15) ──> StandardScaler (Z-score normalization)
   │           └── Categorical Features (17) ──> OneHotEncoder(drop='first')
   │
   ├── 4. Model Training & Benchmarking
   │     ├── Model 1: Linear Regression (Parametric linear baseline)
   │     ├── Model 2: Decision Tree Regressor (Non-parametric partitioning, max_depth=5)
   │     └── Model 3: Random Forest Regressor (Ensemble bagging, 100 estimators)
   │
   ├── 5. Model Evaluation (MAE, MSE, RMSE, R²) on Unseen Test Partition
   │
   └── 6. Champion Model Diagnostics & Interactive Inference Engine
         ├── Actual vs. Predicted & Residual Error Analysis
         └── predict_student_grade() with [0, 20] boundary constraints & advisory tiering
```

---

## 📈 Exploratory Data Analysis: Key Insights

1. **The "Zero-Grade" Anomaly:**  
   $38$ students ($9.62\%$) scored an exact mark of $0$ on their final exam (`G3`). In Portuguese secondary education, this signifies unexcused absence, course abandonment, or disciplinary disqualification.
2. **Academic Momentum:**  
   Prior term marks are the single strongest predictors of terminal performance (`G2`: $r = 0.905$, `G1`: $r = 0.801$).
3. **Study Hours vs. Outcomes:**  
   Increasing weekly study time produces an upward progression in mean grades ($8.85$ for $<2$ hrs/week to $11.26$ for $>10$ hrs/week).
4. **The Attendance Paradox:**  
   School absences exhibit an unexpected near-neutral correlation ($r = +0.034$) because dropouts accumulated fewer recorded school days prior to leaving, whereas diligent passing students recorded excused medical absences.

---

## 🏆 Model Benchmarking & Performance Comparison

All models were evaluated on the held-out test partition ($20\%$, $n = 79$ students) with identical random states (`random_state=42`):

| Rank | Model | MAE (Grade Points) | MSE | RMSE (Grade Points) | Test $R^2$ Score | Train $R^2$ Score | Diagnosis |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Random Forest Regressor** | **1.179** | **3.835** | **1.958** | **0.8130** | 0.9701 | **Champion:** Lowest error, captures non-linear feature interactions, robust generalization. |
| 🥈 | **Linear Regression** | 1.598 | 5.528 | 2.351 | 0.7304 | 0.8447 | Strong linear baseline; effectively captures $G1$/$G2$ momentum. |
| 🥉 | **Decision Tree Regressor** | 1.357 | 6.087 | 2.467 | 0.7031 | 0.9026 | Higher variance and step discretization; sensitive to local splits. |

### Evaluation Metrics Explained
- **Mean Absolute Error (MAE):** Average magnitude of absolute error. Random Forest predicts within **$\approx 1.18$ points** on a $0$–$20$ scale.
- **Root Mean Squared Error (RMSE):** Measures standard deviation of residuals, penalizing large forecasting errors heavily ($1.958$ points).
- **Coefficient of Determination ($R^2$):** Measures proportion of variance explained. The Random Forest model explains **$81.3\%$** of final grade variance on unseen test data.

---

## 🔍 Feature Importance Analysis

From the champion Random Forest model:
1. **`G2` (Second Period Grade):** Accounts for **$79.3\%$** of predictive importance.
2. **`absences`:** Contributes **$11.0\%$** of predictive weight.
3. **`reason_home` & `age`:** Minor contributors ($\approx 1.0\text{–}2.1\%$).
4. **`G1` (First Period Grade):** Provides complementary baseline weighting.

---

## 💻 Interactive Prediction Engine

The notebook implements a robust inference utility `predict_student_grade()` that takes any raw student profile, executes preprocessing, enforces the $[0, 20]$ grade bounds, and returns qualitative academic advisory tiers:

```python
# Predict grade for any student dictionary or DataFrame
result = predict_student_grade(champion_pipeline, custom_student)

print(f"Predicted Final Grade: {result['Predicted_G3']} / 20.00")
print(f"Academic Tier:         {result['Academic_Tier']}")
print(f"Advisory Guidance:     {result['Recommendation']}")
```

### Sample Prediction Outputs

```plaintext
======================================================================
PREDICTION RESULT: STUDENT A (High Engagement Profile)
----------------------------------------------------------------------
Predicted Final Grade (G3): 16.82 / 20.00
Academic Tier:              Excellent (Honor Roll)
Advisory Recommendation:    Maintain current rigorous study routines; eligible for advanced placement or honors distinction.
======================================================================

======================================================================
PREDICTION RESULT: STUDENT B (At-Risk Profile)
----------------------------------------------------------------------
Predicted Final Grade (G3): 6.84 / 20.00
Academic Tier:              Failing (Academic Risk)
Advisory Recommendation:    Immediate academic intervention required. Schedule remedial workshops and advisory consultation.
======================================================================
```

---

## 🚀 How to Run

### Option 1: One-Click Cloud Execution (Google Colab)
Click the badge below to run the notebook in Google Colab with zero local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aryamantomar/Student-Performance-Prediction/blob/main/Student_Performance_Prediction.ipynb)

*The notebook includes an automated dataset fetcher that downloads `student-mat.csv` from UCI automatically if not found in the Colab session.*

---

### Option 2: Local Installation (Jupyter / VS Code)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aryamantomar/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`)*

4. **Launch Jupyter:**
   ```bash
   jupyter notebook Student_Performance_Prediction.ipynb
   ```

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`)
- **Environment:** Jupyter Notebook / Google Colab

---

## ⚠️ Limitations & Future Work

1. **Zero-Grade Discontinuity:** Standard continuous regression models occasionally predict $5$–$8$ for students who suddenly dropped out despite having passing $G1$/$G2$ marks. A **two-stage hurdle model** (binary classification for dropout probability + regression for exam attendees) would improve edge-case accuracy.
2. **Dataset Demographics:** The dataset includes $395$ records from two regional Portuguese secondary schools; cross-institutional validation on larger international cohorts is recommended.
3. **Web Application:** Deploying the pipeline as an interactive counseling tool via Streamlit or Gradio.

---

## 📚 References & Citation

- **Original Study:** Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal.
- **Dataset Source:** [UCI Machine Learning Repository — Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

---

## 👤 Author

**Aryaman Tomar**
- **GitHub:** [@aryamantomar](https://github.com/aryamantomar)
- **Repository:** [Student-Performance-Prediction](https://github.com/aryamantomar/Student-Performance-Prediction)

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
