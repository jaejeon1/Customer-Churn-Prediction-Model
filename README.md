# 📡 Telco Customer Churn Analysis & Prediction Model

> End-to-end data analysis project identifying $3.7M revenue risk

> Building a production-ready churn prediction application

![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-red)
![SHAP](https://img.shields.io/badge/SHAP-ff6b6b)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)

---

<img width="1075" height="467" alt="image" src="https://github.com/user-attachments/assets/ba201a80-4a33-4ec1-ae7b-2a0fa2bcff3b" />
<img width="957" height="313" alt="image" src="https://github.com/user-attachments/assets/ac172cca-5c04-414e-bfd1-7a155af774bf" />

---

## 🔴 The Business Problem

A telecommunications company is losing **26.5% of its customers annually** 
nearly double the industry average of ~15%. 
This translates to **$3.68M in lost revenue** across 1,869 churned customers.

The goal: identify who will churn, why, and what to do about it.

---

## 🔍 Key Findings

| Finding | Insight |
|--------|---------|
| Contract type | Month-to-Month customers churn at **45.8%** vs 2.5% for two-year contracts |
| Early tenure | **53% churn rate** in the first 6 months |
| Fiber Optic (Product) | Fiber Optic Internet users churn at **40.7%** — 5x non-internet customers |
| Protective services | Online Security cuts churn from 31% → **14.6%** |
| Revenue paradox | Churned customers pay **$13 more/month** than those who stay |
| Competitor threat | **46% of revenue loss** ($1.69M) goes directly to competitors |

---

## 📁 Project Structure

```
Customer-Churn-Prediction-Model
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── churn_model.pkl
│   └── feature_cols.pkl
├── data/
│   └── telco.csv
└── notebook/
    └── IBM_telco_churn_analysis.ipynb
```

---

## 📊 Analysis Workflow

Business Problem Definition
↓

Data Cleaning & Validation
→ Missing value investigation (structural vs random)
→ Verified None vs NaN distinction per column
↓

Exploratory Data Analysis
→ Q1: Who churns? (demographics & contract)
→ Q2: Which services drive churn?
→ Q3: What is the financial profile of churners?
↓

Predictive Modelling
→ Initial models flagged 0.99 AUC
→ Data leakage detected: Satisfaction Score (r=0.75)
→ Leaky feature removed, models retrained
→ 4 models compared: Logistic Regression, Decision Tree,
Random Forest, XGBoost
↓

Model Selection & Explainability
→ XGBoost selected: AUC 0.901, stable CV std 0.004
→ SHAP values for business explainability
↓

Production Application
→ Streamlit app for retention team use

---

## 🤖 Model Performance

| Model | CV AUC | Test AUC | Std |
|-------|--------|----------|-----|
| **XGBoost** | **0.8999** | **0.9009** | **0.0040** |
| Logistic Regression | 0.8918 | 0.8921 | 0.0040 |
| Random Forest | 0.8908 | 0.8860 | 0.0041 |
| Decision Tree | 0.8689 | 0.8693 | 0.0101 |

**Final model metrics (XGBoost):**

|  | Precision | Recall | F1 |
|-------|--------|----------|-----|
| Stayed | 0.93 | 0.79 | 0.86 |
| Churned | 0.60 | 0.84 | 0.70 |

<img width="587" height="503" alt="image" src="https://github.com/user-attachments/assets/a721600c-4b1b-48bb-8739-1bb845eb3b5a" />


> Recall of 0.84 means we catch 84% of at-risk customers before they leave.

> Missing a churner costs ~$74/month × remaining tenure.

> A false alarm costs ~$20 retention offer. The model saves significantly more than it costs.

---

## 🔑 Top Churn Drivers (SHAP)

| Feature | Impact |
|---------|--------|
| Contract: Month-to-Month | Highest impact |
| Number of Referrals | Low referrals = high risk |
| Tenure in Months | Early customers most vulnerable |
| Monthly Charge | Higher payers more likely to leave |
| Contract: Two Year | Strong protective effect |


<img width="547" height="377" alt="image" src="https://github.com/user-attachments/assets/49798169-d7e6-4ead-b771-91547cbe0f04" />


---

## 💡 Business Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| 🔴 High | Convert M2M → annual contracts via discount | Reduce churn 18x for converted customers |
| 🔴 High | Early intervention program at months 1-6 | Target 53% churn window |
| 🟡 Medium | Bundle Online Security for Fiber Optic onboarding | Cut churn 31% → 14.6% |
| 🟡 Medium | Launch referral rewards program | Referrals = strongest loyalty signal |
| 🟢 Low | Value communication campaign for high spenders | Address $13/month charge perception gap |

> Scenario: Reducing churn by 5pp → **~$695K recovered annually**

---

## 🚀 Streamlit App

Upload any customer CSV to get instant churn predictions with AI explanations.

**[→ Live App](https://customer-churn-prediction-model-1.streamlit.app/)**

https://customer-churn-prediction-model-1.streamlit.app/

**App Screenshot**
<img width="1083" height="607" alt="image" src="https://github.com/user-attachments/assets/d050e53c-76fe-4cd9-962b-5a2b1c570b35" />


**Features:**
- Upload customer dataset CSV
  
- Instant risk scoring for all customers
  
- Risk segmentation: High / Medium / Low
  
- Individual SHAP explanation per customer
  
- Recommended retention action per customer
  
- Download results as CSV

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core analysis |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
| Scikit-learn | Model training & evaluation |
| XGBoost | Final prediction model |
| SHAP | Model explainability |
| Streamlit | Production application |
| Joblib | Model serialisation |
| GitHub | Version control |

---

## ▶️ Run Locally

```bash
# Clone the repo
git clone https://github.com/jaejeon1/Customer-Churn-Prediction-Model
cd churn-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 50 features 
including demographics, services, contract details, and churn labels.
[Dataset Link](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)

---

## 👤 Author

**Jaewoo Jeon**  
[LinkedIn](www.linkedin.com/in/jaewoo-jeon-5b2585264) · [GitHub](https://github.com/jaejeon1)
