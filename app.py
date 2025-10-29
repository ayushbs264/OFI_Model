# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
from io import BytesIO

# ROOT = Path(__file__).resolve().parents[1]
# DATA_DIR = ROOT / "data"
# MODEL_DIR = ROOT / "models"
# MODEL_PATH = MODEL_DIR / "model.joblib"
# DF_PATH = DATA_DIR / "delivery_performance.csv"

MODEL_PATH = Path("C:/Users/ayush/OneDrive/Desktop/OFI/model.joblib")
DF_PATH = Path("C:/Users/ayush/Downloads/Case study internship data/Data/delivery_performance.csv")

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")
st.title("📦 Predictive Delivery Optimizer")

# -------------------------
# Load resources (cached)
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DF_PATH)
    df.columns = [c.strip() for c in df.columns]
    if {"Actual_Delivery_Days","Promised_Delivery_Days"}.issubset(df.columns):
        df["Delay_Days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
    else:
        df["Delay_Days"] = np.nan
    # Fill missing ratings etc.
    if "Customer_Rating" in df.columns:
        df["Customer_Rating"] = pd.to_numeric(df["Customer_Rating"], errors="coerce")
        df["Customer_Rating"].fillna(df["Customer_Rating"].mean(), inplace=True)
    df["Delivery_Cost_INR"] = pd.to_numeric(df.get("Delivery_Cost_INR"), errors="coerce")
    df["Delivery_Cost_INR"].fillna(df["Delivery_Cost_INR"].median(), inplace=True)
    # Carrier stats
    if "Carrier" in df.columns:
        cs = df.groupby("Carrier").agg(
            carrier_avg_delay_days=("Delay_Days", "mean"),
            carrier_on_time_rate=("Delivery_Status", lambda s: (s.str.lower()=="on-time").mean()),
            carrier_mean_cost=("Delivery_Cost_INR", "mean")
        ).reset_index()
        df = df.merge(cs, on="Carrier", how="left")
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# -------------------------
# Sidebar - filters & nav
# -------------------------
st.sidebar.header("Filters & Data Selection")
sources = ["Delivery Performance"]
selected_sources = st.sidebar.multiselect("Datasets loaded", options=sources, default=sources)

carriers = st.sidebar.multiselect("Carrier(s)", options=sorted(df["Carrier"].unique()), default=sorted(df["Carrier"].unique()))
statuses = st.sidebar.multiselect("Delivery Status", options=sorted(df["Delivery_Status"].unique()), default=sorted(df["Delivery_Status"].unique()))

filtered = df[df["Carrier"].isin(carriers) & df["Delivery_Status"].isin(statuses)]

# quick date filters if there are date columns (optional)
date_cols = [c for c in df.columns if "date" in c.lower()]
if date_cols:
    st.sidebar.write("Date filters (detected):")
    for dc in date_cols:
        st.sidebar.date_input(dc, value=None)

# -------------------------
# KPIs
# -------------------------
st.subheader("Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total deliveries", len(filtered))
c2.metric("On-time rate", f"{(filtered['Delivery_Status'].str.lower()=='on-time').mean()*100:.2f}%")
c3.metric("Avg delay (days)", f"{filtered['Delay_Days'].mean():.2f}")
c4.metric("Avg customer rating", f"{filtered['Customer_Rating'].mean():.2f}")

st.markdown("---")

# -------------------------
# Visualizations (4+)
# -------------------------
st.subheader("Visualizations")

# 1. Average delay by carrier (bar)
fig1 = px.bar(filtered.groupby("Carrier", as_index=False).agg(avg_delay=("Delay_Days","mean")),
              x="Carrier", y="avg_delay", title="Average Delay by Carrier")
st.plotly_chart(fig1, use_container_width=True)

# 2. Delivery status distribution (pie)
fig2 = px.pie(filtered, names="Delivery_Status", title="Delivery Status Distribution", hole=0.3)
st.plotly_chart(fig2, use_container_width=True)

# 3. Cost vs Rating (scatter)
if {"Delivery_Cost_INR","Customer_Rating"}.issubset(filtered.columns):
    fig3 = px.scatter(filtered, x="Delivery_Cost_INR", y="Customer_Rating", color="Delivery_Status", title="Delivery Cost vs Customer Rating", hover_data=["Carrier"])
    st.plotly_chart(fig3, use_container_width=True)

# 4. Delay distribution by quality issue (box)
if "Quality_Issue" in filtered.columns:
    fig4 = px.box(filtered, x="Quality_Issue", y="Delay_Days", title="Delay by Quality Issue", points="all")
    st.plotly_chart(fig4, use_container_width=True)

# Optional: map if latitude/longitude or distance present
if "Distance_km" in filtered.columns:
    st.map(filtered.dropna(subset=["Distance_km"]).assign(lat=lambda d: d["Distance_km"]*0.0 + 28.6, lon=lambda d: d["Distance_km"]*0.0 + 77.2)[["lat","lon"]])  # placeholder: requires real lat lon

st.markdown("---")

# -------------------------
# Predictive interface
# -------------------------
st.subheader("Predict Delay for a New Order")
st.markdown("Enter pre-delivery info (these are features available before the delivery)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    in_carrier = st.selectbox("Carrier", options=sorted(df["Carrier"].unique()))
with col2:
    in_promised = st.number_input("Promised Delivery Days", min_value=0, max_value=60, value=3)
with col3:
    in_cost = st.number_input("Estimated Delivery Cost (INR)", min_value=0.0, max_value=100000.0, value=500.0, step=10.0)
with col4:
    in_expected_rating = st.slider("Expected Customer Rating", 1.0, 5.0, 4.0, 0.1)

# carrier stats lookup (history)
row = df[df["Carrier"]==in_carrier]
carrier_avg_delay = float(row["carrier_avg_delay_days"].mean()) if not row.empty else 0.0
carrier_on_time_rate = float(row["carrier_on_time_rate"].mean()) if not row.empty else 0.5
carrier_mean_cost = float(row["carrier_mean_cost"].mean()) if not row.empty else in_cost

if st.button("Predict Delay Risk"):
    if model is None:
        st.error("Model not found. Run train_model.py to produce models/model.joblib.")
    else:
        X_new = pd.DataFrame([{
            "Promised_Delivery_Days": in_promised,
            "Delivery_Cost_INR": in_cost,
            "Carrier": in_carrier,
            "carrier_avg_delay_days": carrier_avg_delay,
            "carrier_on_time_rate": carrier_on_time_rate,
            "carrier_mean_cost": carrier_mean_cost
        }])
        prob = model.predict_proba(X_new)[:,1][0] if hasattr(model.named_steps["clf"], "predict_proba") else None
        pred = model.predict(X_new)[0]
        st.metric("Predicted delay probability", f"{prob*100:.2f}%" if prob is not None else "n/a")
        st.write("Predicted label:", "Delayed" if pred==1 else "On-Time")
        # Simple rule-based corrective actions
        st.subheader("Suggested corrective actions")
        if prob is None:
            st.info("Probability not available; showing rule-based suggestions.")
        if prob is not None and prob < 0.25:
            st.success("Low risk — standard handling.")
            st.write("- Normal scheduling. No need to escalate.")
        elif prob is not None and prob < 0.6:
            st.warning("Medium risk — monitor & prioritize.")
            st.write("- Prioritize packing, assign better vehicle if available.")
            st.write("- Contact operations to monitor.")
        else:
            st.error("High risk — urgent actions recommended.")
            st.write("- Consider alternate faster carrier / express service.")
            st.write("- Move pickup earlier or allocate priority resources.")
            st.write("- Proactively inform customer and offer compensations if appropriate.")
        # minimal explainability: show feature values
        st.write("Prediction inputs:")
        st.json(X_new.to_dict(orient="records")[0])

st.markdown("---")

# -------------------------
# Data export and EDA table
# -------------------------
st.subheader("Data Explorer & Export")
st.write("View and export the filtered dataset below.")
st.dataframe(filtered.head(200))
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_delivery_data.csv", mime="text/csv")


