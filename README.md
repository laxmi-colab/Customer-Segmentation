# Customer Segmentation Using Machine Learning

A Streamlit application that groups mall customers using **K-Means clustering** and presents customer profiles and evaluation plots.

## Features
- Loads the Mall Customers CSV dataset (or accepts an uploaded CSV).
- Checks required columns, missing values and duplicate records.
- Uses Annual Income (k$) and Spending Score (1-100) as clustering features.
- Scales features using StandardScaler.
- Compares K values using the Elbow Method and Silhouette Score.
- Displays customer clusters and descriptive cluster profiles.
- Assigns a new customer to the nearest learned cluster.
- Downloads customer assignments and cluster profiles.

## Folder structure
```text
Customer_Segmentation_ML_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.txt
├── outputs/
│   └── .gitkeep
└── src/
    ├── __init__.py
    └── segmentation.py
```

## Requirements
- Python 3.10 or later
- Internet connection for first-time default dataset download, or a local CSV uploaded in the app

## Run on Windows
Open VS Code terminal in this project folder and run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py` is not recognized, use `python` instead:
```bash
python -m venv .venv
```

The browser opens at `http://localhost:8501`.

## Dataset
The app uses the public Mall Customers educational dataset. The application downloads it from:
https://raw.githubusercontent.com/rithwik00/Mall-Customer-Dataset/main/Mall_Customers.csv

Expected columns:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

`CustomerID` is an identifier and is not used as a clustering feature. The baseline uses Annual Income and Spending Score.

## Algorithm
K-Means is an unsupervised clustering algorithm. It assigns each record to the nearest centroid and repeatedly updates centroids. StandardScaler is applied before distance-based clustering. The Elbow curve shows inertia for candidate K values; the Silhouette Score helps assess separation.

## GitHub upload
Create an empty public repository on GitHub named `Customer-Segmentation-Using-Machine-Learning`, then run:

```bash
git init
git add .
git commit -m "Initial customer segmentation project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Customer-Segmentation-Using-Machine-Learning.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not upload `.venv` or secrets.

## Responsible use
This is an educational decision-support prototype. Cluster labels are arbitrary and results depend on the dataset, features, scaling and K. Do not use cluster membership as the sole basis for decisions about individuals.
