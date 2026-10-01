import streamlit as st

# -----------------------------
# LOGIN CONFIGURATION
# -----------------------------
VALID_USERS = {
    "admin": "admin123",
    "sanika": "sanika11"
    "aditya": "aditya11"
    "vishwajit": "vishwajit11"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Secure Login")
    st.subheader("Network Traffic Analytics Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.rerun()

# -----------------------------
# LOGIN CHECK
# -----------------------------
if not st.session_state.logged_in:
    login()
    st.stop()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Network Traffic Analytics Dashboard", layout="wide")

st.title("📊 Network Traffic Data Analytics")
st.caption("Data Analysis + Pattern Analysis")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("KDDTrain+.txt", header=None)

data = load_data()
st.success("Dataset Loaded Successfully")

# -----------------------------
# Preprocessing
# -----------------------------
encoder = LabelEncoder()
for col in [1, 2, 3]:
    data[col] = encoder.fit_transform(data[col])

data[41] = data[41].apply(lambda x: 0 if x == "normal" else 1)

X = data.drop(41, axis=1)
y = data[41]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.markdown("### 👤 User")
st.sidebar.write(f"Logged in as: **{st.session_state.user}**")

if st.sidebar.button("Logout"):
    logout()

st.sidebar.header("⚙️ Analysis Controls")

threshold_percent = st.sidebar.slider(
    "Analysis Threshold (%)",
    min_value=10,
    max_value=50,
    value=30
)

n_estimators = st.sidebar.slider(
    "Number of Trees",
    min_value=50,
    max_value=300,
    value=200
)

# -----------------------------
# Train Model
# -----------------------------
model = IsolationForest(
    n_estimators=n_estimators,
    contamination=0.45,
    random_state=42
)

model.fit(X_scaled)
scores = model.decision_function(X_scaled)
threshold = np.percentile(scores, threshold_percent)
pred = np.where(scores < threshold, 1, 0)

# -----------------------------
# ANALYTICS METRICS
# -----------------------------
st.subheader("📊 Data Analytics Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(pred))
col2.metric("Anomalous Records", int(np.sum(pred)))
col3.metric("Normal Records", int(len(pred) - np.sum(pred)))
col4.metric("Anomaly %", f"{(np.sum(pred)/len(pred))*100:.2f}%")

st.caption(f"🕒 Scan Time: {datetime.now()}")

# -----------------------------
# Data Pattern Summary
# -----------------------------
st.subheader("📋 Data Pattern Summary (Sample)")

alert_df = pd.DataFrame({
    "Connection": range(1, 11),
    "Status": ["⚠️ Unusual Pattern" if scores[i] < threshold else "✅ Normal Pattern" for i in range(10)],
    "Score": scores[:10]
})

st.dataframe(alert_df, use_container_width=True)

# -----------------------------
# PCA Visualization
# -----------------------------
st.subheader("📈 Traffic Pattern Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[pred == 0, 0], X_pca[pred == 0, 1],
           c="green", alpha=0.3, s=5, label="Normal")
ax.scatter(X_pca[pred == 1, 0], X_pca[pred == 1, 1],
           c="red", alpha=0.6, s=10, label="Anomaly")

ax.set_title("Network Traffic Pattern Analysis")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Download Analytical Results
# -----------------------------
st.subheader("📁 Analytical Report")

anomalies = data[pred == 1].copy()
anomalies["anomaly_score"] = scores[pred == 1]

csv = anomalies.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Analyzed Records CSV",
    data=csv,
    file_name="analyzed_records.csv",
    mime="text/csv"
)

st.success("Data Analytics Dashboard Running Successfully 🚀")
