import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

REQUIRED_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]
DATA_URL = "https://raw.githubusercontent.com/rithwik00/Mall-Customer-Dataset/main/Mall_Customers.csv"

def prepare_data(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    work = df.copy().drop_duplicates()
    for col in REQUIRED_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=REQUIRED_COLUMNS)
    if len(work) < 3:
        raise ValueError("At least three valid customer records are required.")
    return work

def run_segmentation(df, k=5):
    clean = prepare_data(df)
    if k >= len(clean):
        raise ValueError("K must be smaller than the number of valid customer records.")
    X = clean[REQUIRED_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows = []
    for candidate_k in range(2, min(11, len(clean))):
        candidate = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
        labels = candidate.fit_predict(X_scaled)
        rows.append({
            "k": candidate_k,
            "inertia": candidate.inertia_,
            "silhouette": silhouette_score(X_scaled, labels)
        })
    metrics = pd.DataFrame(rows)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clean["Cluster"] = model.fit_predict(X_scaled)
    profiles = clean.groupby("Cluster")[REQUIRED_COLUMNS].agg(
        ["count", "mean", "median", "min", "max"]
    )
    score = silhouette_score(X_scaled, clean["Cluster"])
    return {
        "data": clean,
        "metrics": metrics,
        "profiles": profiles,
        "scaler": scaler,
        "model": model,
        "silhouette": score
    }

def assign_customer(scaler, model, income, spending_score):
    new_data = pd.DataFrame([{
        REQUIRED_COLUMNS[0]: income,
        REQUIRED_COLUMNS[1]: spending_score
    }])
    scaled = scaler.transform(new_data[REQUIRED_COLUMNS])
    return int(model.predict(scaled)[0])
