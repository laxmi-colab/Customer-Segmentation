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


# Customer Segmentation Using Machine Learning

## 1. Project Title
**Customer Segmentation Using Machine Learning**

## 2. Team Members

| S. No. | Name | Roll Number |
|---|---|---|
| 1 | S. Keshava Reddy | 25RBAIM0001 |
| 2 | Ch. Laxmi Prasanna | 25RBAIM0079 |

## 3. Project Description

Customer Segmentation Using Machine Learning is an interactive application that groups mall customers based on their annual income and spending score.

This project uses the K-Means clustering algorithm, an unsupervised machine learning technique, to identify different customer groups. It helps understand customer purchasing behaviour and provides useful business insights.

The application includes dataset analysis, data visualization, model evaluation, cluster profiles, and prediction for new customers.

## 4. Dataset Used

- **Dataset Name:** Mall Customers Dataset
- **CSV File:** `Mall_Customers.csv`
- **Number of Records:** 200 customers
- **Number of Columns:** 5

### Dataset Features

1. CustomerID
2. Gender
3. Age
4. Annual Income (k$)
5. Spending Score (1-100)

**Dataset Source:** [Kaggle – Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

Place the downloaded CSV file inside the `data` folder.

```text
data/Mall_Customers.csv
```

## 5. Algorithms and Models Used

- **K-Means Clustering:** Groups customers into different clusters based on their features.
- **StandardScaler:** Standardizes the selected numerical features before clustering.
- **Elbow Method:** Helps examine a suitable number of clusters using inertia.
- **Silhouette Score:** Evaluates how well the customer groups are separated.

The model uses Annual Income and Spending Score as its main clustering features. CustomerID is used only as an identifier.

## 6. Technologies and Libraries Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 7. Project Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/seelamkeshavareddy1-art/Customer-Segmentation-Using-Machine-Learning.git
```

### Step 2: Open the Project Folder

```bash
cd Customer-Segmentation-Using-Machine-Learning
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

## 8. Results

The application successfully analyzed 200 customer records and generated five customer clusters.

### Model Evaluation Results

| Metric | Result |
|---|---:|
| Total Customers | 200 |
| Selected Number of Clusters (K) | 5 |
| Silhouette Score | 0.555 |
| Number of Clusters Generated | 5 |
| Duplicate Rows | 0 |

### Cluster Distribution

| Cluster | Number of Customers |
|---|---:|
| Cluster 0 | 81 |
| Cluster 1 | 39 |
| Cluster 2 | 22 |
| Cluster 3 | 35 |
| Cluster 4 | 23 |

### Application Features

- Dataset overview and preview
- Missing value and duplicate row analysis
- Elbow Method visualization
- Silhouette Score comparison
- Customer segmentation scatter plot
- Cluster profile analysis
- New customer cluster prediction
- Download customer segment results
- Download cluster profiles

The results show that customers can be grouped into different segments based on income and spending behaviour. The cluster IDs are labels, not rankings.

## 9. Conclusion

This project demonstrates how machine learning can be used to understand customer behaviour through clustering. The K-Means algorithm identifies customer groups based on annual income and spending score. The interactive Streamlit dashboard makes it easier to explore the dataset, visualize customer segments, evaluate the model, and predict the cluster of a new customer.

The project provides a practical example of applying unsupervised machine learning to a business problem.