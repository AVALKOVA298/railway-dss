import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Railway DSS",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================
delay_model = joblib.load("delay_model.pkl")
conflict_model = joblib.load("conflict_model.pkl")

# =========================
# WARM LIGHT UI STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fffaf4 0%, #ffffff 55%, #fff7ed 100%);
    color: #162033;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
    border-right: 1px solid #fed7aa;
}

section[data-testid="stSidebar"] * {
    color: #162033 !important;
}

section[data-testid="stSidebar"] label {
    color: #162033 !important;
    font-weight: 600 !important;
}

h1 {
    color: #162033;
    font-size: 44px !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #162033;
    font-weight: 700 !important;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #fed7aa;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 6px 18px rgba(180, 83, 9, 0.08);
    min-height: 112px;
}

.metric-label {
    color: #64748b;
    font-size: 15px;
    font-weight: 600;
}

.metric-value-blue {
    color: #2563eb;
    font-size: 32px;
    font-weight: 800;
}

.metric-value-orange {
    color: #ea580c;
    font-size: 32px;
    font-weight: 800;
}

.metric-value-green {
    color: #15803d;
    font-size: 32px;
    font-weight: 800;
}

.metric-value-red {
    color: #dc2626;
    font-size: 32px;
    font-weight: 800;
}

.recommendation-low {
    background: #ecfdf3;
    border: 1px solid #bbf7d0;
    color: #15803d;
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 17px;
    font-weight: 600;
}

.recommendation-medium {
    background: #fff7ed;
    border: 1px solid #fdba74;
    color: #c2410c;
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 17px;
    font-weight: 600;
}

.recommendation-high {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    color: #b91c1c;
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 17px;
    font-weight: 600;
}

.scheme-box {
    background: #ffffff;
    border: 1.5px solid #fdba74;
    border-radius: 18px;
    padding: 24px 28px;
    margin-top: 8px;
    margin-bottom: 24px;
    box-shadow: 0 6px 18px rgba(180, 83, 9, 0.08);
}

.scheme-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    color: #162033;
    font-size: 19px;
    font-weight: 700;
    white-space: nowrap;
}

.station-line {
    flex: 1;
    text-align: center;
}

.scheme-conflict {
    color: #dc2626;
    font-weight: 800;
}

.scheme-note {
    margin-top: 16px;
    color: #dc2626;
    font-size: 15px;
    font-weight: 600;
    text-align: center;
}

.risk-info-box {
    background: #ffffff;
    border: 1px solid #fed7aa;
    border-left: 5px solid #ea580c;
    border-radius: 12px;
    padding: 13px 16px;
    color: #475569;
    font-size: 14px;
    margin-top: 16px;
    margin-bottom: 18px;
}

.footer-box {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 14px;
    padding: 16px 20px;
    color: #475569;
    font-size: 14px;
    margin-top: 24px;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #fed7aa !important;
    color: #162033 !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: #162033 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("Railway Decision Support System")
st.subheader("Train Delay and Crossing Conflict Prediction")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("⚙️ Operational Parameters")

train_type = st.sidebar.selectbox(
    "Train Type",
    ["Freight", "Container", "Passenger"]
)

train_type_map = {
    "Freight": 0,
    "Container": 1,
    "Passenger": 2
}
train_type_value = train_type_map[train_type]

mass_t = st.sidebar.slider(
    "Train Mass, tons",
    1000,
    9000,
    3500
)

length_wagons = st.sidebar.slider(
    "Number of Wagons",
    10,
    100,
    35
)

priority = st.sidebar.selectbox(
    "Train Priority",
    ["Low", "Medium", "High"]
)

priority_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}
priority_value = priority_map[priority]

current_delay_min = st.sidebar.slider(
    "Current Delay, min",
    0,
    120,
    5
)

opposite_delay_min = st.sidebar.slider(
    "Opposing Train Delay, min",
    0,
    120,
    2
)

load_pct = st.sidebar.slider(
    "Infrastructure Load, %",
    40,
    100,
    60
)

headway_following_min = st.sidebar.slider(
    "Following Headway, min",
    1,
    60,
    33
)

segment_length_km = st.sidebar.slider(
    "Segment Length, km",
    5,
    100,
    11
)

technical_speed_kmh = st.sidebar.slider(
    "Technical Speed, km/h",
    30,
    120,
    70
)

temperature_c = st.sidebar.slider(
    "Temperature, °C",
    -40,
    40,
    5
)

snow_fog_label = st.sidebar.selectbox(
    "Snow / Fog",
    ["No", "Yes"]
)
snow_fog = 1 if snow_fog_label == "Yes" else 0

# =========================
# DERIVED FEATURES
# =========================
travel_time_min = round((segment_length_km / technical_speed_kmh) * 60, 2)

prev_delay_min = max(0, current_delay_min - 4)
mean_last3_delay_min = round(
    (current_delay_min + prev_delay_min + opposite_delay_min) / 3,
    2
)

critical_interval = 1 if headway_following_min <= 15 else 0
priority_conflict = 1 if priority_value >= 1 and opposite_delay_min >= 8 else 0
conflicts_last2h = 3 if load_pct >= 90 else 2 if load_pct >= 75 else 1
night_window = 0
precipitation = snow_fog

# =========================
# MODEL INPUT
# =========================
input_df = pd.DataFrame([{
    "train_type": train_type_value,
    "mass_t": mass_t,
    "length_wagons": length_wagons,
    "priority": priority_value,
    "segment_id": 3,
    "segment_length_km": segment_length_km,
    "slope_permille": 8,
    "station_tracks": 3,
    "hour": 14,
    "day_of_week": 3,
    "month": 5,
    "night_window": night_window,
    "temperature_c": temperature_c,
    "precipitation": precipitation,
    "snow_fog": snow_fog,
    "current_delay_min": current_delay_min,
    "load_pct": load_pct,
    "headway_following_min": headway_following_min,
    "opposite_delay_min": opposite_delay_min,
    "prev_delay_min": prev_delay_min,
    "mean_last3_delay_min": mean_last3_delay_min,
    "conflicts_last2h": conflicts_last2h,
    "travel_time_min": travel_time_min,
    "critical_interval": critical_interval,
    "priority_conflict": priority_conflict,
    "time_index": 25000,
    "historical_avg_wagons": 64,
    "historical_avg_weight": 5200
}])

# =========================
# REAL MODEL PREDICTIONS
# =========================
model_delay = float(delay_model.predict(input_df)[0])
model_conflict_probability = float(conflict_model.predict_proba(input_df)[0][1])

# =========================
# RESPONSIVE OPERATIONAL LAYER
# =========================
risk_score = 0.0
risk_score += 0.22 if current_delay_min >= 30 else 0.08 if current_delay_min >= 10 else 0.00
risk_score += 0.20 if opposite_delay_min >= 20 else 0.08 if opposite_delay_min >= 5 else 0.00
risk_score += 0.25 if load_pct >= 85 else 0.12 if load_pct >= 70 else 0.03
risk_score += 0.20 if headway_following_min <= 15 else 0.08 if headway_following_min <= 30 else 0.00
risk_score += 0.10 if snow_fog == 1 else 0.00
risk_score += 0.07 if mass_t >= 6500 else 0.00

conflict_probability = min(
    0.99,
    max(0.02, 0.55 * model_conflict_probability + 0.45 * risk_score)
)

operational_delay = (
    current_delay_min * 1.4
    + opposite_delay_min * 0.45
    + max(load_pct - 50, 0) * 0.35
    + snow_fog * 5
    + max(20 - headway_following_min, 0) * 0.5
)

predicted_delay = round(0.65 * model_delay + 0.35 * operational_delay, 1)
predicted_delay = max(0, predicted_delay)

# =========================
# RISK LEVEL RANGES
# =========================
# LOW:    0.00–0.39
# MEDIUM: 0.40–0.69
# HIGH:   0.70–1.00

if conflict_probability >= 0.70:
    risk_level = "HIGH"
    recommendation = (
        "Hold the lower-priority train at the previous siding and prioritize "
        "the higher-priority train."
    )
    rec_class = "recommendation-high"
    risk_color_class = "metric-value-red"
elif conflict_probability >= 0.40:
    risk_level = "MEDIUM"
    recommendation = (
        "Monitor the situation closely and prepare local timetable adjustment if delay increases."
    )
    rec_class = "recommendation-medium"
    risk_color_class = "metric-value-orange"
else:
    risk_level = "LOW"
    recommendation = "No operational intervention required."
    rec_class = "recommendation-low"
    risk_color_class = "metric-value-green"

# =========================
# KPI CARDS
# =========================
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Delay</div>
        <div class="metric-value-blue">{predicted_delay:.1f} min</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Conflict Probability</div>
        <div class="metric-value-orange">{conflict_probability:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk Level</div>
        <div class="{risk_color_class}">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="risk-info-box">
Risk level ranges: LOW = 0.00–0.39, MEDIUM = 0.40–0.69, HIGH = 0.70–1.00.
</div>
""", unsafe_allow_html=True)

# =========================
# DISPATCHER RECOMMENDATION
# =========================
st.markdown("### Dispatcher Recommendation")
st.markdown(f"""
<div class="{rec_class}">
    {recommendation}
</div>
""", unsafe_allow_html=True)

# =========================
# OPERATIONAL SCHEME
# =========================
st.markdown("### Railway Operational Scheme")

st.markdown("""
<div class="scheme-box">
    <div class="scheme-line">
        <div>🚆 Train A</div>
        <div class="station-line">—— Station A —— Station B —— <span class="scheme-conflict">Crossing Point</span> —— Station D —— Station E ——</div>
        <div>🚆 Train B</div>
    </div>
    <div class="scheme-note">
        Potential crossing conflict zone
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# DYNAMIC TIMETABLE GRAPH
# =========================
st.markdown("### Recommended Timetable Adjustment")

stations = [1, 2, 3, 4, 5]

base_shift = min(max(predicted_delay * 0.08, 2), 10)
risk_shift = 3 if risk_level == "LOW" else 7 if risk_level == "MEDIUM" else 12

train_a_before_x = [
    0,
    20,
    40 + current_delay_min * 0.08,
    60 + current_delay_min * 0.10,
    80 + current_delay_min * 0.12
]

train_b_before_x = [
    80 + opposite_delay_min * 0.10,
    60 + opposite_delay_min * 0.08,
    40 + opposite_delay_min * 0.05,
    20,
    0
]

train_a_after_x = [
    train_a_before_x[0],
    train_a_before_x[1] + base_shift * 0.25,
    train_a_before_x[2] + base_shift * 0.45,
    train_a_before_x[3] + base_shift * 0.40,
    train_a_before_x[4] + base_shift * 0.25
]

train_b_after_x = [
    train_b_before_x[0] + risk_shift,
    train_b_before_x[1] + risk_shift * 0.75,
    train_b_before_x[2] + risk_shift * 0.45,
    train_b_before_x[3] + risk_shift * 0.25,
    train_b_before_x[4] + risk_shift * 0.10
]

fig, ax = plt.subplots(figsize=(12, 5.8))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#fffaf4")

ax.plot(
    train_a_before_x,
    stations,
    marker="o",
    linewidth=2.6,
    color="#2563EB",
    label="Train A BEFORE"
)

ax.plot(
    train_b_before_x,
    stations,
    marker="o",
    linewidth=2.6,
    color="#DC2626",
    label="Train B BEFORE"
)

ax.plot(
    train_a_after_x,
    stations,
    "--",
    marker="s",
    linewidth=2.6,
    color="#16A34A",
    label="Train A AFTER"
)

ax.plot(
    train_b_after_x,
    stations,
    "--",
    marker="s",
    linewidth=2.6,
    color="#F97316",
    label="Train B AFTER"
)

ax.set_title(
    "Timetable Lines Before and After Crossing Conflict Adjustment",
    fontsize=15,
    color="#162033"
)

ax.set_xlabel("Time, min")
ax.set_ylabel("Station")

ax.set_yticks(stations)
ax.set_yticklabels([
    "Station A",
    "Station B",
    "Crossing Point",
    "Station D",
    "Station E"
])

ax.grid(True, alpha=0.25)
ax.legend(frameon=True)

ax.annotate(
    "Predicted conflict",
    xy=(train_a_before_x[2], 3),
    xytext=(train_a_before_x[2] + 10, 4),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10
)

ax.annotate(
    "Adjusted timetable",
    xy=(train_b_after_x[2], 3),
    xytext=(train_b_after_x[2] + 12, 2),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10
)

st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer-box">
    This prototype supports dispatchers in predicting train delays, detecting crossing conflicts,
    and generating local timetable adjustment recommendations for overloaded single-track railway sections.
</div>
""", unsafe_allow_html=True)