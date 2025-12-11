# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle
from pathlib import Path

# ---------- Page ----------
st.set_page_config(page_title="AQI Prediction APP", layout="centered")
st.markdown(
    """
    <style>
      .stApp { max-width: 750px; margin: 0 auto; }
      h1 { font-size: 2.4rem !important; font-weight: 800; margin-bottom: 1.2rem; }
      .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("AQI Prediction APP")

# ---------- Load Model ----------
MODEL_PATH = Path(__file__).with_name("knn_pipe.pkl")
with open(MODEL_PATH, "rb") as f:
    try:
        model = joblib.load(f)
    except Exception:
        f.seek(0)
        model = pickle.load(f)

# ---------- Cities (your 26) ----------
cities = [
    "Ahmedabad","Aizawl","Amaravati","Amritsar","Bengaluru","Bhopal","Brajrajnagar",
    "Chandigarh","Chennai","Coimbatore","Delhi","Ernakulam","Gurugram","Guwahati",
    "Hyderabad","Jaipur","Jorapokhar","Kochi","Kolkata","Lucknow","Mumbai",
    "Patna","Shillong","Talcher","Thiruvananthapuram","Visakhapatnam"
]

# ---------- Expected columns ----------
def expected_cols(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            if hasattr(step, "transformers_"):
                cols = []
                for _, _, spec in step.transformers_:
                    if isinstance(spec, (list, tuple, np.ndarray, pd.Index)):
                        cols.extend(list(spec))
                if cols:
                    return list(cols)
    return ["City","PM2.5","NO","NO2","NOX","CO","SO2","O3","Benzene"]

EXP = expected_cols(model)

# ---------- UI ----------
city = st.selectbox("City", cities, index=cities.index("Chennai") if "Chennai" in cities else 0)
pm25 = st.number_input("PM2.5",  value=0.00, min_value=0.00, step=0.01, format="%.2f")
no   = st.number_input("NO",     value=0.00, min_value=0.00, step=0.01, format="%.2f")
no2  = st.number_input("NO2",    value=0.00, min_value=0.00, step=0.01, format="%.2f")
nox  = st.number_input("NOX",    value=0.00, min_value=0.00, step=0.01, format="%.2f")
co   = st.number_input("CO",     value=0.00, min_value=0.00, step=0.01, format="%.2f")
so2  = st.number_input("SO2",    value=0.00, min_value=0.00, step=0.01, format="%.2f")
o3   = st.number_input("O3",     value=0.00, min_value=0.00, step=0.01, format="%.2f")
benz = st.number_input("Benzene",value=0.00, min_value=0.00, step=0.01, format="%.2f")

# ---------- Build row aligned to model ----------
alias = {
    "City": ["City","city","LOCATION","Location"],
    "PM2.5": ["PM2.5","PM2_5","PM25","pm2_5","pm25"],
    "NO": ["NO","no"],
    "NO2": ["NO2","no2"],
    "NOX": ["NOX","nox"],
    "CO": ["CO","co"],
    "SO2": ["SO2","so2"],
    "O3": ["O3","o3"],
    "Benzene": ["Benzene","benzene","C6H6"],
}
vals = {"City": city, "PM2.5": pm25, "NO": no, "NO2": no2, "NOX": nox, "CO": co, "SO2": so2, "O3": o3, "Benzene": benz}

def build_row(expected):
    row = {}
    for col in expected:
        setval = None
        for k, al in alias.items():
            if col in al:
                setval = vals[k] if k == "City" else float(vals[k])
                break
        if setval is None:
            setval = "" if col.lower() in ["city","location","state","region"] else 0.0
        row[col] = setval
    X = pd.DataFrame([row], columns=expected)
    for c in X.columns:
        if not isinstance(X[c].iloc[0], str):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

# ---------- Predict (Bucket + Code ID) ----------
if st.button("Predict AQIBucket"):
    X = build_row(EXP)
    pred = model.predict(X)[0]  # predicted label (str or int)

    # derive code id robustly
    code_id = None
    bucket = str(pred)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        try:
            # if numeric classes (e.g., 0..5)
            if isinstance(pred, (int, np.integer)) or (isinstance(pred, str) and pred.isdigit()):
                code_id = int(pred)
                # if there is a matching class label for this index, show it
                if 0 <= code_id < len(classes) and not str(classes[code_id]).isdigit():
                    bucket = str(classes[code_id])
            else:
                code_id = classes.index(pred) if pred in classes else None
        except Exception:
            pass
    if code_id is None:
        mapping = {"Good":0,"Satisfactory":1,"Moderate":2,"Poor":3,"Very Poor":4,"Severe":5}
        code_id = mapping.get(bucket, "")

    st.success(f"AQI Bucket :{bucket}")


