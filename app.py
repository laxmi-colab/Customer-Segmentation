import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.segmentation import run_segmentation, assign_customer, REQUIRED_COLUMNS, DATA_URL

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ Customer Segmentation Using Machine Learning")
st.caption("An interactive customer grouping and business insight system using K-Means clustering.")

@st.cache_data
def load_default_data():
    local = Path("data/Mall_Customers.csv")
    if local.exists():
        return pd.read_csv(local)
    return pd.read_csv(DATA_URL)

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload Mall Customers CSV", type=["csv"])
try:
    df = pd.read_csv(uploaded) if uploaded else load_default_data()
except Exception as exc:
    st.error("Could not load the default dataset. Check your internet connection or upload Mall_Customers.csv.")
    st.code(str(exc))
    st.stop()

st.subheader("1. Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total customers", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Duplicate rows", int(df.duplicated().sum()))
with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)
st.write("**Missing values by column**")
st.dataframe(df.isna().sum().rename("Missing values"), use_container_width=True)

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    st.error(f"Required columns missing: {missing_cols}. Please upload a compatible CSV.")
    st.stop()

st.sidebar.header("Clustering Settings")
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)
try:
    result = run_segmentation(df, k)
except Exception as exc:
    st.error(f"Unable to run clustering: {exc}")
    st.stop()

clean = result["data"]
metrics = result["metrics"]
profiles = result["profiles"]
st.subheader("2. Model Evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Selected clusters (K)", k)
m2.metric("Silhouette score", f'{result["silhouette"]:.3f}')
m3.metric("Customers analyzed", len(clean))

left, right = st.columns(2)
with left:
    st.markdown("**Elbow Method**")
    fig, ax = plt.subplots()
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.grid(alpha=.25)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Silhouette Score**")
    fig, ax = plt.subplots()
    ax.plot(metrics["k"], metrics["silhouette"], marker="o")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    ax.grid(alpha=.25)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("3. Customer Segments")
fig, ax = plt.subplots()
for cluster_id in sorted(clean["Cluster"].unique()):
    part = clean[clean["Cluster"] == cluster_id]
    ax.scatter(part[REQUIRED_COLUMNS[0]], part[REQUIRED_COLUMNS[1]],
               label=f"Cluster {cluster_id}", alpha=.75)
ax.set_xlabel(REQUIRED_COLUMNS[0])
ax.set_ylabel(REQUIRED_COLUMNS[1])
ax.set_title(f"Customer Segments (K={k})")
ax.legend()
ax.grid(alpha=.2)
st.pyplot(fig)
plt.close(fig)

st.subheader("4. Cluster Profiles")
st.dataframe(profiles.round(2), use_container_width=True)
st.dataframe(clean, use_container_width=True)
st.download_button("Download customer segment results",
                   clean.to_csv(index=False).encode("utf-8"),
                   "customer_segments.csv", "text/csv")
st.download_button("Download cluster profiles",
                   profiles.to_csv().encode("utf-8"),
                   "cluster_profiles.csv", "text/csv")

st.subheader("5. Assign a New Customer")
st.write("Enter annual income and spending score to assign a customer to the nearest learned cluster.")
col1, col2 = st.columns(2)
income = col1.number_input("Annual Income (k$)", min_value=0.0, max_value=500.0, value=60.0, step=1.0)
score = col2.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
if st.button("Predict Customer Segment", type="primary"):
    cluster_id = assign_customer(result["scaler"], result["model"], income, score)
    st.success(f"This customer is assigned to Cluster {cluster_id}.")
    st.info("Cluster IDs are labels, not rankings. Interpret the result using the cluster profile above.")

st.caption("Educational decision-support prototype. Customer segments are exploratory and should not be used as the sole basis for decisions about individuals.")
