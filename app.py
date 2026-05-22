import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #EEF3F8; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1100px; }

/* ── Top header ── */
.app-header {
    background: linear-gradient(130deg, #1a3f6b 0%, #2563a8 55%, #3d8fd4 100%);
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 30px rgba(26,63,107,0.22);
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #fff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.2px;
}
.app-header p { color: rgba(255,255,255,0.78); font-size: 0.88rem; margin: 0; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.6rem;
}

/* ── Section card ── */
.section-card {
    background: #fff;
    border-radius: 14px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.1rem;
    border: 1px solid #DDE8F2;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #2563a8;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #EEF3F8;
}

/* ── Result ── */
.result-box {
    border-radius: 14px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-safe { background: linear-gradient(135deg,#E8F5E9,#F4FFF5); border: 2px solid #43A047; }
.result-risk { background: linear-gradient(135deg,#FFEBEE,#FFF6F6); border: 2px solid #E53935; }
.result-icon { font-size: 2.8rem; margin-bottom: 0.4rem; }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin-bottom: 0.2rem;
}
.result-safe .result-title { color: #2E7D32; }
.result-risk .result-title { color: #C62828; }
.result-sub { font-size: 0.85rem; color: #666; font-weight: 500; }

/* ── Prob bar ── */
.prob-wrap {
    background: #EEF3F8;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin: 0.9rem 0 0.25rem;
}
.prob-fill-safe { height:100%; border-radius:99px; background: linear-gradient(90deg,#43A047,#76C442); }
.prob-fill-risk { height:100%; border-radius:99px; background: linear-gradient(90deg,#E53935,#FF6B35); }
.prob-labels { display:flex; justify-content:space-between; font-size:0.7rem; color:#999; margin-top:2px; }

/* ── Metric tiles ── */
.tiles { display:flex; gap:0.7rem; flex-wrap:wrap; margin-bottom:1rem; }
.tile {
    flex:1; min-width:72px;
    background:#F5F9FD;
    border:1px solid #DDE8F2;
    border-radius:10px;
    padding:0.75rem 0.5rem;
    text-align:center;
}
.tile-val { font-size:1.05rem; font-weight:700; color:#1a3f6b; }
.tile-lbl { font-size:0.65rem; color:#999; text-transform:uppercase; letter-spacing:0.7px; margin-top:2px; }

/* ── Input summary ── */
.summary-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; }
.summary-row { display:flex; justify-content:space-between; align-items:center;
    font-size:0.8rem; padding:0.4rem 0.6rem; border-radius:7px; background:#F5F9FD; }
.summary-key { color:#666; font-weight:500; }
.summary-val { color:#1a3f6b; font-weight:700; }

/* ── Recommendation ── */
.rec-item { display:flex; gap:0.7rem; align-items:flex-start;
    padding:0.65rem 0.8rem; border-radius:9px; margin-bottom:0.5rem;
    background:#F5F9FD; border-left:3px solid #2563a8; font-size:0.83rem; color:#333; }
.rec-icon { font-size:1rem; flex-shrink:0; margin-top:1px; }

/* ── Disclaimer ── */
.disclaimer {
    background:#FFF8E1; border:1px solid #FFD54F; border-radius:10px;
    padding:0.8rem 1rem; font-size:0.78rem; color:#6D4C00; margin-top:0.8rem;
}

/* ── Streamlit overrides ── */
div[data-testid="stSlider"] > div { padding-top: 0.2rem; }
label[data-testid="stWidgetLabel"] > div { font-size: 0.83rem !important; font-weight: 600; color: #334; }
div[data-baseweb="select"] { border-radius: 9px !important; }
.stSelectbox > div > div { border-radius: 9px !important; }
div[data-testid="stNumberInput"] input { border-radius: 9px !important; }
.stButton > button {
    background: linear-gradient(130deg, #1a3f6b, #2563a8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(37,99,168,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/heart_disease_best_model.joblib")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ── Helper functions ──────────────────────────────────────────────────────────
def classify_bp(ap_hi, ap_lo):
    if ap_hi >= 140 or ap_lo >= 90:    return "Hypertension Stage 2"
    elif ap_hi >= 130 or ap_lo >= 80:  return "Hypertension Stage 1"
    elif ap_hi >= 120 and ap_lo < 80:  return "Elevated"
    else:                              return "Normal"

def classify_bmi(bmi):
    if bmi < 18.5:  return "Underweight"
    elif bmi < 25:  return "Normal"
    elif bmi < 30:  return "Overweight"
    else:           return "Obesity"

def get_recommendations(data, prediction, proba):
    recs = []
    if data["ap_hi"] >= 130 or data["ap_lo"] >= 80:
        recs.append(("🩺", "Your blood pressure is elevated. Consult a doctor and consider reducing sodium intake."))
    if data["bmi"] >= 25:
        recs.append(("⚖️", "Your BMI is above normal. Regular physical activity and a balanced diet can help."))
    if data["cholesterol"] != "Normal":
        recs.append(("🥗", "Elevated cholesterol detected. Include fibre-rich foods and reduce saturated fats."))
    if data["gluc"] != "Normal":
        recs.append(("🍬", "Blood glucose is above normal. Limit sugar intake and schedule a glucose test."))
    if data["smoke"] == "Yes":
        recs.append(("🚭", "Smoking significantly increases cardiovascular risk. Consider a quit-smoking programme."))
    if data["alco"] == "Yes":
        recs.append(("🍷", "Alcohol consumption raises heart disease risk. Aim to reduce or eliminate intake."))
    if data["active"] == "Not Active":
        recs.append(("🏃", "Physical inactivity is a major risk factor. Aim for 30 minutes of moderate exercise daily."))
    if data["age_years"] >= 50 and prediction == 1:
        recs.append(("📅", "Age is a non-modifiable risk factor. Regular cardiovascular screenings are recommended."))
    if not recs:
        recs.append(("✅", "Your lifestyle and clinical inputs look healthy. Keep maintaining good habits!"))
    return recs

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="badge">🫀 ML Classification Project</div>
  <h1>Heart Disease Prediction</h1>
  <p>Enter patient clinical and lifestyle details below to predict cardiovascular disease risk .</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model file. Make sure **heart_disease_best_model.joblib** is in the same folder as this app.\n\n`{model_error}`")
    st.stop()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

# ════════════════════════════════════════════════════════
#  LEFT COLUMN – Inputs
# ════════════════════════════════════════════════════════
with left:

    # ── Personal Information ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age      = st.number_input("Age (years)", min_value=5, max_value=100, value=25, step=1)
        height   = st.number_input("Height (cm)", min_value=120, max_value=220, value=168, step=1)
    with c2:
        gender   = st.selectbox("Gender", ["Male", "Female"])
        weight   = st.number_input("Weight (kg)", min_value=25, max_value=250, value=72, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Blood Pressure & Clinical ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Blood Pressure & Clinical Measurements</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        ap_hi       = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=280, value=120, step=1,
                                       help="Upper blood pressure number e.g. 120 in 120/80")
        cholesterol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"])
    with c4:
        ap_lo       = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=160, value=80, step=1,
                                       help="Lower blood pressure number e.g. 80 in 120/80")
        gluc        = st.selectbox("Glucose Level", ["Low", "Normal", "High"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Lifestyle ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏃 Lifestyle Factors</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        smoke  = st.selectbox("Smoking",  ["No", "Yes"])
    with c6:
        alco   = st.selectbox("Alcohol",  ["No", "Yes"])
    with c7:
        active = st.selectbox("Physical Activity", ["Active", "Not Active"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ──
    predict_clicked = st.button("🔍  Predict Heart Disease Risk")

# ════════════════════════════════════════════════════════
#  RIGHT COLUMN – Results
# ════════════════════════════════════════════════════════
with right:

    if predict_clicked:

        # ── Derived features ──
        bmi   = round(weight / (height / 100) ** 2, 2)
        pp    = ap_hi - ap_lo
        map_  = round((ap_hi + 2 * ap_lo) / 3, 2)
        bp_cat  = classify_bp(ap_hi, ap_lo)
        bmi_cat = classify_bmi(bmi)

        # ── Build DataFrame ──
        input_df = pd.DataFrame([{
            "age_years":              age,
            "height":                 height,
            "weight":                 weight,
            "ap_hi":                  ap_hi,
            "ap_lo":                  ap_lo,
            "bmi":                    bmi,
            "pulse_pressure":         pp,
            "mean_arterial_pressure": map_,
            "gender":                 gender,
            "cholesterol":            cholesterol,
            "gluc":                   gluc,
            "smoke":                  smoke,
            "alco":                   alco,
            "active":                 active,
            "bp_category":            bp_cat,
            "bmi_category":           bmi_cat,
        }])

        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        risk_pct   = round(proba[1] * 100, 1)
        safe_pct   = round(proba[0] * 100, 1)

        # ── Result box ──
        if pred == 0:
            st.markdown(f"""
            <div class="result-box result-safe">
              <div class="result-icon">✅</div>
              <div class="result-title">Low Risk</div>
              <div class="result-sub">No cardiovascular disease detected</div>
              <div class="prob-wrap"><div class="prob-fill-safe" style="width:{safe_pct}%"></div></div>
              <div class="prob-labels"><span>0%</span><span>Confidence: {safe_pct}%</span><span>100%</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-risk">
              <div class="result-icon">⚠️</div>
              <div class="result-title">High Risk</div>
              <div class="result-sub">Cardiovascular disease detected</div>
              <div class="prob-wrap"><div class="prob-fill-risk" style="width:{risk_pct}%"></div></div>
              <div class="prob-labels"><span>0%</span><span>Confidence: {risk_pct}%</span><span>100%</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Computed metrics tiles ──
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Computed Clinical Values</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="tiles">
          <div class="tile"><div class="tile-val">{bmi}</div><div class="tile-lbl">BMI</div></div>
          <div class="tile"><div class="tile-val">{pp}</div><div class="tile-lbl">Pulse Pressure</div></div>
          <div class="tile"><div class="tile-val">{map_}</div><div class="tile-lbl">Mean Art. Pressure</div></div>
          <div class="tile"><div class="tile-val">{bmi_cat}</div><div class="tile-lbl">BMI Category</div></div>
          <div class="tile"><div class="tile-val" style="font-size:0.75rem">{bp_cat}</div><div class="tile-lbl">BP Category</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Input summary ──
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-grid">
          <div class="summary-row"><span class="summary-key">Age</span><span class="summary-val">{age} yrs</span></div>
          <div class="summary-row"><span class="summary-key">Gender</span><span class="summary-val">{gender}</span></div>
          <div class="summary-row"><span class="summary-key">Height</span><span class="summary-val">{height} cm</span></div>
          <div class="summary-row"><span class="summary-key">Weight</span><span class="summary-val">{weight} kg</span></div>
          <div class="summary-row"><span class="summary-key">Systolic BP</span><span class="summary-val">{ap_hi} mmHg</span></div>
          <div class="summary-row"><span class="summary-key">Diastolic BP</span><span class="summary-val">{ap_lo} mmHg</span></div>
          <div class="summary-row"><span class="summary-key">Cholesterol</span><span class="summary-val">{cholesterol}</span></div>
          <div class="summary-row"><span class="summary-key">Glucose</span><span class="summary-val">{gluc}</span></div>
          <div class="summary-row"><span class="summary-key">Smoking</span><span class="summary-val">{smoke}</span></div>
          <div class="summary-row"><span class="summary-key">Alcohol</span><span class="summary-val">{alco}</span></div>
          <div class="summary-row"><span class="summary-key">Activity</span><span class="summary-val">{active}</span></div>
          <div class="summary-row"><span class="summary-key">Risk %</span><span class="summary-val" style="color:{'#C62828' if pred==1 else '#2E7D32'}">{risk_pct}%</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Recommendations ──
        recs = get_recommendations(
            {"ap_hi": ap_hi, "ap_lo": ap_lo, "bmi": bmi,
             "cholesterol": cholesterol, "gluc": gluc,
             "smoke": smoke, "alco": alco, "active": active, "age_years": age},
            pred, proba
        )
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Recommendations</div>', unsafe_allow_html=True)
        for icon, text in recs:
            st.markdown(f'<div class="rec-item"><span class="rec-icon">{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Disclaimer ──
        st.markdown("""
        <div class="disclaimer">
          ⚠️ <strong>Medical Disclaimer:</strong> This prediction is generated by a machine learning model
          for educational purposes only. It is not a substitute for professional medical advice, diagnosis,
          or treatment. Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Placeholder before prediction ──
        st.markdown("""
        <div class="section-card" style="text-align:center; padding: 3rem 1.5rem;">
          <div style="font-size:3.5rem; margin-bottom:1rem;">🫀</div>
          <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#1a3f6b; margin-bottom:0.5rem;">
            Ready to Predict
          </div>
          <div style="font-size:0.85rem; color:#888; max-width:280px; margin:0 auto; line-height:1.6;">
            Fill in the patient details on the left and click
            <strong style="color:#2563a8;">Predict Heart Disease Risk</strong>
            to see the result.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model info card ──
        st.markdown("""
        <div class="section-card">
          <div class="section-title">⚙️ Model Information</div>
          <div class="tiles">
            <div class="tile"><div class="tile-val">RF</div><div class="tile-lbl">Algorithm</div></div>
            <div class="tile"><div class="tile-val">160</div><div class="tile-lbl">Estimators</div></div>
            <div class="tile"><div class="tile-val">8</div><div class="tile-lbl">Max Depth</div></div>
            <div class="tile"><div class="tile-val">0.819</div><div class="tile-lbl">ROC-AUC</div></div>
            <div class="tile"><div class="tile-val">16</div><div class="tile-lbl">Features</div></div>
          </div>
          <div style="font-size:0.78rem; color:#888; margin-top:0.3rem;">
            Tuned Random Forest · StandardScaler + OneHotEncoder pipeline · Trained on cardiovascular disease dataset
          </div>
        </div>
        """, unsafe_allow_html=True)
