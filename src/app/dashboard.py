"""Streamlit demo. Run: streamlit run src/app/dashboard.py

Loads models/pipeline.joblib, predicts risk, and draws a local SHAP waterfall for the entered patient.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.explain.attribution import transform

st.title("GlucoSense-AI: undiagnosed diabetes screening")
st.caption("Research demo. Not medical advice.")

path = Path("models/pipeline.joblib")
if not path.exists():
    st.warning("Train a model and save it to models/pipeline.joblib first.")
    st.stop()
pipe = joblib.load(path)

c1, c2 = st.columns(2)
row = {
    "age": c1.number_input("Age", 18, 100, 45),
    "sex": c1.selectbox("Sex", ["male", "female"]),
    "bmi": c1.number_input("BMI", 12.0, 70.0, 27.0),
    "waist_cm": c1.number_input("Waist (cm)", 50.0, 200.0, 95.0),
    "height_cm": c2.number_input("Height (cm)", 120.0, 220.0, 172.0),
    "systolic_bp": c2.number_input("Systolic BP", 70, 250, 125),
    "diastolic_bp": c2.number_input("Diastolic BP", 40, 150, 80),
    "family_history_diabetes": c2.selectbox("Family history", ["no", "yes"]),
    "smoker": c2.selectbox("Smoker", ["no", "yes"]),
    "race_eth": "3",
}
X = pd.DataFrame([row])
risk = pipe.predict_proba(X)[0, 1]
st.metric("Predicted risk", f"{risk:.1%}")

explainer = shap.TreeExplainer(pipe.named_steps["model"])
Z = transform(pipe, X)
sv = explainer.shap_values(Z)[0]
exp = shap.Explanation(values=sv, base_values=explainer.expected_value, data=Z.iloc[0].values,
                       feature_names=list(Z.columns))
shap.plots.waterfall(exp, show=False)
st.pyplot(plt.gcf())
