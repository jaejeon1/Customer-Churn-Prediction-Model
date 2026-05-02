import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Load model assets ─────────────────────────────────────────
import shap
import xgboost as xgb

model      = joblib.load('churn_model.pkl')
feat_cols  = joblib.load('feature_cols.pkl')

# Recreate explainer from model directly
explainer  = shap.TreeExplainer(model)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telco Customer Churn Predictor")
st.markdown("Upload a customer CSV file to predict churn risk and get AI-powered explanations.")

# ── Helper: preprocess uploaded df ───────────────────────────
def preprocess(df):
    binary_cols = [
        'Online Security', 'Premium Tech Support', 'Online Backup',
        'Device Protection Plan', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Married', 'Paperless Billing', 'Senior Citizen'
    ]
    multi_cols = ['Contract', 'Internet Type', 'Payment Method', 'Offer']

    processed = df.copy()

    # Binary encoding
    for col in binary_cols:
        if col in processed.columns:
            processed[col] = processed[col].map({'Yes': 1, 'No': 0})

    # One hot encode
    processed = pd.get_dummies(processed, columns=multi_cols, drop_first=False)

    # Drop leaky + non-feature columns
    drop_cols = ['Customer ID', 'Satisfaction Score', 'Churn', 
                 'Churn Label', 'Churn Score', 'Churn Category', 
                 'Churn Reason', 'Customer Status', 'Gender',
                 'Country', 'State', 'City', 'Zip Code',
                 'Latitude', 'Longitude', 'Population', 'Quarter',
                 'Phone Service', 'Multiple Lines', 'Internet Service',
                 'Under 30', 'Dependents', 'Number of Dependents',
                 'Referred a Friend', 'Offer',
                 'Avg Monthly Long Distance Charges',
                 'Avg Monthly GB Download', 'Unlimited Data',
                 'Total Charges', 'Total Refunds',
                 'Total Extra Data Charges',
                 'Total Long Distance Charges', 'Total Revenue', 'CLTV']
    
    for col in drop_cols:
        if col in processed.columns:
            processed = processed.drop(columns=[col])

    # Align to training feature columns
    processed = processed.reindex(columns=feat_cols, fill_value=0)

    return processed

# ── Helper: risk label ────────────────────────────────────────
def risk_label(prob):
    if prob >= 0.7:
        return "🔴 HIGH", "red"
    elif prob >= 0.4:
        return "🟡 MEDIUM", "orange"
    else:
        return "🟢 LOW", "green"

# ── Helper: top SHAP reasons ──────────────────────────────────
def get_top_reasons(shap_vals, feature_names, n=3):
    pairs = list(zip(feature_names, shap_vals))
    sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    
    reasons = []
    for feat, val in sorted_pairs[:n]:
        direction = "↑ increases churn risk" if val > 0 else "↓ reduces churn risk"
        # Clean feature name
        clean = feat.replace('_', ' ').replace('Contract ', 'Contract: ')\
                    .replace('Internet Type ', 'Internet: ')\
                    .replace('Payment Method ', 'Payment: ')\
                    .replace('Offer ', 'Offer: ')
        reasons.append((clean, val, direction))
    return reasons

# ── Helper: recommended action ────────────────────────────────
def get_action(reasons, prob):
    top_feat = reasons[0][0].lower()
    
    if prob < 0.4:
        return "✅ Low risk. Standard engagement is sufficient."
    elif 'month-to-month' in top_feat or 'contract' in top_feat:
        return "💡 Offer discounted annual contract upgrade immediately."
    elif 'referral' in top_feat:
        return "💡 Enrol in referral rewards program to boost engagement."
    elif 'tenure' in top_feat:
        return "💡 Trigger early onboarding check-in and loyalty offer."
    elif 'security' in top_feat or 'tech support' in top_feat:
        return "💡 Bundle protective services at discounted rate."
    elif 'fiber' in top_feat:
        return "💡 Proactive outreach — benchmark offer vs competitors."
    elif 'charge' in top_feat or 'monthly' in top_feat:
        return "💡 Schedule value review call — communicate service worth."
    else:
        return "💡 Flag for retention team outreach within 7 days."

# ── File uploader ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload customer CSV file", 
    type=['csv'],
    help="Use the original telco dataset format"
)

if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.success(f"✅ {len(raw_df)} customers loaded")

    # Preprocess
    try:
        processed = preprocess(raw_df)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Predict
    probs  = model.predict_proba(processed)[:, 1]
    preds  = model.predict(processed)

    # SHAP
    shap_vals = explainer.shap_values(processed)

    # Build results table
    results = pd.DataFrame({
        'Customer ID':       raw_df['Customer ID'] if 'Customer ID' in raw_df.columns else range(len(raw_df)),
        'Churn Probability': (probs * 100).round(1),
        'Risk Level':        [risk_label(p)[0] for p in probs],
        'Top Reason':        [get_top_reasons(shap_vals[i], feat_cols, n=1)[0][0] 
                              for i in range(len(probs))]
    }).sort_values('Churn Probability', ascending=False)

    # ── Summary metrics ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",  len(raw_df))
    col2.metric("High Risk 🔴",     int((probs >= 0.7).sum()))
    col3.metric("Medium Risk 🟡",   int(((probs >= 0.4) & (probs < 0.7)).sum()))
    col4.metric("Low Risk 🟢",      int((probs < 0.4).sum()))

    st.markdown("---")

    # ── Risk table ────────────────────────────────────────────
    st.subheader("Customer Risk Rankings")
    st.dataframe(results, use_container_width=True, height=300)

    # ── Download button ───────────────────────────────────────
    csv_out = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_out,
        file_name='churn_predictions.csv',
        mime='text/csv'
    )

    st.markdown("---")

    # ── Individual customer deep dive ─────────────────────────
    st.subheader("Individual Customer Analysis")

    customer_ids = results['Customer ID'].astype(str).tolist()
    selected_id  = st.selectbox("Select a customer to analyse:", customer_ids)

    # Find index
    if 'Customer ID' in raw_df.columns:
        idx = raw_df[raw_df['Customer ID'].astype(str) == selected_id].index[0]
    else:
        idx = int(selected_id)

    prob     = probs[idx]
    label, color = risk_label(prob)
    reasons  = get_top_reasons(shap_vals[idx], feat_cols, n=3)
    action   = get_action(reasons, prob)

    # Customer info + prediction
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown(f"### Customer: `{selected_id}`")
        st.markdown(f"**Churn Probability:** {prob*100:.1f}%")
        st.markdown(f"**Risk Level:** :{color}[{label}]")
        st.markdown("---")
        st.markdown("**Top 3 Drivers:**")
        for i, (feat, val, direction) in enumerate(reasons):
            st.markdown(f"{i+1}. **{feat}** — {direction}")
        st.markdown("---")
        st.markdown(f"**Recommended Action:**")
        st.info(action)

    with c2:
        st.markdown("**Customer Profile:**")
        
        # Pull raw customer data for context
        customer_row = raw_df[raw_df['Customer ID'].astype(str) == selected_id].iloc[0]
        
        profile_data = {
            'Contract':        customer_row.get('Contract', 'N/A'),
            'Tenure':          f"{customer_row.get('Tenure in Months', 'N/A')} months",
            'Internet Type':   customer_row.get('Internet Type', 'N/A'),
            'Monthly Charge':  f"${customer_row.get('Monthly Charge', 'N/A')}",
            'Senior Citizen':  customer_row.get('Senior Citizen', 'N/A'),
            'Online Security': customer_row.get('Online Security', 'N/A'),
            'Referrals':       customer_row.get('Number of Referrals', 'N/A'),
        }
        
      

else:
    st.info("👆 Upload a CSV file to get started.")
    st.markdown("""
    **Expected columns include:**
    - `Customer ID`, `Contract`, `Tenure in Months`
    - `Internet Type`, `Monthly Charge`, `Senior Citizen`
    - `Online Security`, `Premium Tech Support`
    - and other standard telco fields
    """)
