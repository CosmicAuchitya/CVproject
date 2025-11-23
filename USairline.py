import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="USA Airline Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
)

# -------------------------------------------------
# Custom CSS – dark theme + glass effect + input styles
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background: radial-gradient(circle at top left, #1e293b 0, #020617 45%, #000000 100%);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar glass effect */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.98));
        border-right: 1px solid rgba(148,163,184,0.25);
        backdrop-filter: blur(18px);
        box-shadow: 8px 0 30px rgba(0,0,0,0.55);
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        color: #e5e7eb !important;
    }

    /* Generic label text (including above inputs) */
    label {
        color: #e5e7eb !important;
        font-weight: 500;
        font-size: 0.88rem;
    }

    /* All input, select, textarea boxes: white background + black text */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 0.75rem !important;
        border: 1px solid #e5e7eb !important;
        padding: 0.35rem 0.75rem !important;
    }

    /* Placeholder text inside inputs */
    input::placeholder, textarea::placeholder {
        color: #6b7280 !important;
    }

    /* Number input arrows color fix (Chrome) */
    input[type=number] {
        -moz-appearance: textfield;
    }
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    /* Selectbox arrow background stay white */
    select {
        background-image: none !important;
    }

    /* Slider label + value */
    [data-baseweb="slider"] p {
        color: #e5e7eb !important;
    }

    /* Main cards – glass effect */
    .glass-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(30,64,175,0.6));
        border-radius: 1.5rem;
        padding: 1.75rem 2.25rem;
        border: 1px solid rgba(148,163,184,0.28);
        box-shadow: 0 28px 80px rgba(15,23,42,0.9);
        backdrop-filter: blur(22px);
    }

    .glass-card-soft {
        background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(15,23,42,0.92));
        border-radius: 1.25rem;
        padding: 1.5rem 2rem;
        border: 1px solid rgba(51,65,85,0.8);
        box-shadow: 0 24px 60px rgba(15,23,42,0.85);
        backdrop-filter: blur(18px);
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: radial-gradient(circle at top left, #4f46e5, #0f172a);
        color: #e5e7eb;
        border: 1px solid rgba(129,140,248,0.6);
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        letter-spacing: 0.03em;
        color: #f9fafb;
    }

    .subtitle {
        color: #cbd5f5;
        font-size: 0.95rem;
        max-width: 38rem;
    }

    /* Predict button styling */
    .stButton>button {
        border-radius: 999px;
        padding: 0.65rem 1.9rem;
        font-weight: 600;
        font-size: 0.97rem;
        border: none;
        background: radial-gradient(circle at 0 0, #4f46e5, #7c3aed);
        color: #f9fafb;
        box-shadow: 0 18px 40px rgba(88,80,236,0.65);
    }
    .stButton>button:hover {
        filter: brightness(1.06);
        box-shadow: 0 22px 60px rgba(88,80,236,0.85);
    }

    /* Info / warning / success blocks width tweak */
    .stAlert {
        border-radius: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Load trained ensemble models
# -------------------------------------------------
@st.cache_resource
def load_models():
    models = load("dt_ensemble_models.joblib")
    return models


ensemble_models = load_models()


def predict_delay_with_proba(df: pd.DataFrame):
    """
    Returns:
      class_pred: 0 or 1  (0 = on-time, 1 = delayed)
      avg_proba:  probability of delay (0–1)
    """
    proba_list = []
    for model in ensemble_models:
        proba = model.predict_proba(df)[:, 1]  # P(delay=1)
        proba_list.append(proba)

    proba_stack = np.column_stack(proba_list)
    avg_proba = np.mean(proba_stack, axis=1)

    class_pred = (avg_proba >= 0.5).astype(int)  # standard 50% threshold
    return class_pred, avg_proba


# -------------------------------------------------
# Sidebar – input form
# -------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ Flight Inputs")

    DayOfWeek = st.selectbox(
        "Day of Week (1 = Mon, 7 = Sun)",
        [1, 2, 3, 4, 5, 6, 7],
        index=5,
    )

    Time = st.number_input(
        "Scheduled Departure Time (HHMM)",
        min_value=0,
        max_value=2359,
        value=1430,
        step=10,
    )

    Length = st.number_input(
        "Flight Duration (minutes)",
        min_value=10,
        max_value=1000,
        value=120,
    )

    Hour = st.slider(
        "Departure Hour",
        min_value=0,
        max_value=23,
        value=14,
    )

    Airline_Age = st.number_input(
        "Airline Age (years)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
    )

    Founded = st.number_input(
        "Airline Founded Year",
        min_value=1900,
        max_value=2100,
        value=1998,
    )

    src_elevation_ft = st.number_input(
        "Source Airport Elevation (ft)",
        min_value=-1000.0,
        max_value=20000.0,
        value=200.0,
    )

    dest_elevation_ft = st.number_input(
        "Destination Airport Elevation (ft)",
        min_value=-1000.0,
        max_value=20000.0,
        value=150.0,
    )

    src_n_runways = st.number_input(
        "Source Airport Runways",
        min_value=1,
        max_value=20,
        value=3,
    )

    dest_n_runways = st.number_input(
        "Destination Airport Runways",
        min_value=1,
        max_value=20,
        value=2,
    )

    Airline = st.text_input(
        "Airline Code (e.g. UA, AA, DL)",
        value="UA",
        placeholder="Enter airline code",
    )

    AirportFrom = st.text_input(
        "Source Airport Code (e.g. JFK)",
        value="JFK",
        placeholder="Enter source airport",
    )

    AirportTo = st.text_input(
        "Destination Airport Code (e.g. LAX)",
        value="LAX",
        placeholder="Enter destination airport",
    )

    src_Hub_Type = st.selectbox(
        "Source Hub Type",
        ["Hub", "Non-Hub"],
        index=0,
    )

    dest_Hub_Type = st.selectbox(
        "Destination Hub Type",
        ["Hub", "Non-Hub"],
        index=0,
    )

    predict_clicked = st.button("Predict Delay")


# -------------------------------------------------
# Main layout
# -------------------------------------------------
col_main, col_side = st.columns([2.4, 1])

with col_main:
    st.markdown(
        """
        <div class="glass-card">
            <div class="pill-badge">USA Airlines · ML App</div>
            <h1 class="main-title">USA Airline Flight Delay Predictor</h1>
            <p class="subtitle">
                Estimate the probability of delay for a US domestic flight using a decision tree ensemble model.
                Enter the flight details in the sidebar and click <strong>Predict Delay</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_side:
    st.markdown(
        """
        <div class="glass-card-soft">
            <h3 style="margin-top:0; margin-bottom:0.75rem;">ℹ️ About</h3>
            <p style="font-size:0.9rem; color:#cbd5f5;">
                This app uses a machine learning model trained on the
                <strong>US Airlines delay</strong> dataset to estimate whether
                a flight will be delayed.
            </p>
            <ul style="font-size:0.85rem; color:#e5e7eb; padding-left:1.1rem; margin-bottom:0;">
                <li>Fill in flight details on the left</li>
                <li>Click <strong>Predict Delay</strong></li>
                <li>See delay probability & status</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# Prediction result card
st.markdown('<div class="glass-card-soft">', unsafe_allow_html=True)
st.markdown("### Prediction Result")

if not predict_clicked:
    st.info("Enter values on the left and click **Predict Delay**.")
else:
    # Build input DataFrame
    input_df = pd.DataFrame(
        [
            {
                "DayOfWeek": DayOfWeek,
                "Time": Time,
                "Length": Length,
                "Hour": Hour,
                "Airline_Age": Airline_Age,
                "Founded": Founded,
                "src_elevation_ft": src_elevation_ft,
                "dest_elevation_ft": dest_elevation_ft,
                "src_n_runways": src_n_runways,
                "dest_n_runways": dest_n_runways,
                "Airline": Airline,
                "AirportFrom": AirportFrom,
                "AirportTo": AirportTo,
                "src_Hub_Type": src_Hub_Type,
                "dest_Hub_Type": dest_Hub_Type,
            }
        ]
    )

    class_pred, avg_proba = predict_delay_with_proba(input_df)
    pred_class = int(class_pred[0])
    delay_proba = float(avg_proba[0])
    delay_pct = round(delay_proba * 100, 1)

    st.write(f"**Estimated chance of delay:** `{delay_pct}%`")
    st.caption(
        "Standard rule: If delay probability ≥ 50%, the flight is treated as **likely delayed**."
    )

    if pred_class == 1:
        st.error(
            f"🚨 Based on the inputs, this flight is **LIKELY TO BE DELAYED**.\n\n"
            f"The model estimates about **{delay_pct}%** chance of delay."
        )
    else:
        st.success(
            f"✅ Based on the inputs, this flight is **LIKELY TO BE ON-TIME**.\n\n"
            f"The model estimates only **{delay_pct}%** chance of delay."
        )

st.markdown("</div>", unsafe_allow_html=True)
