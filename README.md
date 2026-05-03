# 📡 Telco Customer Churn Analysis & Prediction Model

> End-to-end data analysis project dientifying $3.7M revenue risk
> Building a production-ready churn prediction application

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

> Recall of 0.84 means we catch 84% of at-risk customers before they leave.

> Missing a churner costs ~$74/month × remaining tenure.

> A false alarm costs ~$20 retention offer. The model saves significantly more than it costs.

---

## 🔑 Top Churn Drivers (SHAP)

Contract: Month-to-Month   Highest impact

Number of Referrals        Low referrals = high risk

Tenure in Months           Early customers most vulnerable

Monthly Charge             Higher payers more likely to leave

Contract: Two Year         Strong protective effect

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

**[→ Live App]([http://192.168.1.214:8501](https://customer-churn-prediction-model-1.streamlit.app/))**

**App Screenshot**
<img width="1118" height="846" alt="image" src="https://github.com/user-attachments/assets/daa0435b-2b7a-46e5-9e5b-2e6ec71f4045" />



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
streamlit run app/app.py
```

---

## 📁 Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 50 features 
including demographics, services, contract details, and churn labels.

---

## 👤 Author

**Jaewoo Jeon**  
[LinkedIn](www.linkedin.com/in/jaewoo-jeon-5b2585264) · [GitHub](https://github.com/jaejeon1)
