# 🏥 Medicaid Fraud Detection — Virginia
### HHS / DOGE Data Release Analysis

This project is for **educational and research purposes only**. Flagged providers are **statistical anomalies**, not confirmed fraud cases. No conclusions should be drawn about any specific provider without further investigation by qualified professionals.

A Jupyter Notebook + Plotly Dash project for exploring and detecting potential fraud patterns in the **2026 DOGE/HHS Medicaid provider spending dataset**, focused on the state of Ohio.

---

## 📋 Table of Contents
1. [Background](#background)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [How to Download the Data](#how-to-download-the-data)
5. [Running the Notebook](#running-the-notebook)
6. [Running the Dashboard](#running-the-dashboard)
7. [Fraud Detection Algorithms](#fraud-detection-algorithms)
8. [Database Schema](#database-schema)
9. [Contributing](#contributing)

---

## 📖 Background

In **February 2026**, HHS (via DOGE) released an unprecedented **227+ million row dataset** of Medicaid provider spending covering years **2018–2024**. The data includes billing provider NPIs, procedure codes (HCPCS), claim counts, recipient counts, and total paid amounts.

This project:
- Downloads and loads the **Virginia** subset of that data
- Stores it in a **local MySQL database**
- Runs **4 fraud detection algorithms** against it
- Presents findings in an **interactive Plotly Dash dashboard**

**Primary data source:** https://opendata.hhs.gov/datasets/medicaid-provider-spending  
**Virginia pre-parsed file:** https://getmedicaiddata.com

---

## 📁 Project Structure

```
medicaid-fraud-detection/
├── .gitignore                  ← Keeps credentials & data off GitHub
├── README.md                   ← You are here
├── requirements.txt            ← All Python dependencies
├── config.ini.example          ← Template for your MySQL credentials
├── config.ini                  ← YOUR credentials (never committed)
├── data/                       ← Drop your Virginia CSV here (never committed)
│   └── .gitkeep                ← Keeps the empty folder in Git
├── notebooks/
│   └── medicaid_fraud_analysis.ipynb   ← Main analysis notebook
└── dashboard/
    └── app.py                  ← Plotly Dash dashboard
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/medicaid-fraud-detection.git
cd medicaid-fraud-detection
```

### 2. Install Python Dependencies
Make sure you have Python 3.10+ installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. Configure Your MySQL Credentials
```bash
# Copy the example config file
copy config.ini.example config.ini   # Windows
```
Then open `config.ini` in any text editor and replace `YOUR_PASSWORD_HERE` with your actual MySQL root password.

### 4. Create the MySQL Database
Open **MySQL Workbench** or **Command Prompt** and run:
```sql
CREATE DATABASE medicaid_fraud;
```
The notebook will handle creating all tables automatically.

### 5. Download the Virginia Data
See the next section ↓

---

## 📥 How to Download the Data

The Virginia CSV is available from a third-party site that pre-parsed the full HHS dataset by state and enriched it with provider name/address data from the NPPES NPI Registry.

1. Go to **https://getmedicaiddata.com**
2. Scroll down to find **"Download Virginia File"**
3. Click the Google Drive link — it will download a `.zip` file
4. **Unzip** the file — you'll get a `.csv` file inside
5. **Rename** that CSV to `VA_medicaid.csv`
6. **Move** it into the `data/` folder in this project

> ⚠️ The `data/` folder is in `.gitignore` — your data file will never be accidentally pushed to GitHub.

**Can't find Virginia yet?** The site is uploading states in batches. If Virginia isn't listed yet:
- You can download the **full 10GB national file** from https://opendata.hhs.gov/datasets/medicaid-provider-spending and the notebook will filter it to Ohio for you.
- Or check back at getmedicaiddata.com — Virginia is a large state and should be available soon.

---

## 🚀 Running the Notebook

```bash
cd notebooks
jupyter notebook medicaid_fraud_analysis.ipynb
```

Run cells **top to bottom**. The notebook will:
1. Verify your data file exists
2. Load and profile the Virginia data
3. Connect to MySQL and create tables
4. Load data into the database
5. Run all 4 fraud detection algorithms
6. Save flagged results back to MySQL

---

## 📊 Running the Dashboard

After running the notebook (so the database is populated), open a **new Command Prompt** window and run:

```bash
cd dashboard
python app.py
```

Then open your browser and go to: **http://127.0.0.1:8050**

The dashboard will let you toggle between all 4 fraud detection algorithms and explore flagged providers interactively.

---

## 🔍 Fraud Detection Algorithms

| Algorithm | What it detects | Best for |
|---|---|---|
| **Z-Score / IQR** | Providers whose billing amounts are statistically far from the mean | Simple, explainable outliers |
| **Isolation Forest** | Anomalies across multiple features simultaneously using ML | Complex, multi-dimensional fraud patterns |
| **DBSCAN Clustering** | Providers that don't fit into any natural cluster | Unusual billing behavior vs. peer groups |
| **Benford's Law** | Leading digit distributions that deviate from natural patterns | Fabricated or manipulated billing amounts |

---

## 🗄️ Database Schema

```sql
-- Raw provider claims data
CREATE TABLE claims ( ... );

-- Providers flagged by each algorithm
CREATE TABLE fraud_flags ( ... );

-- Summary statistics per provider
CREATE TABLE provider_summary ( ... );
```
*(Full schema auto-generated by the notebook)*

---

## 🤝 Contributing

Pull requests welcome. Please ensure `config.ini` and any CSV files are excluded from commits.

---

## ⚖️ Disclaimer

This project is for **educational and research purposes only**. Flagged providers are **statistical anomalies**, not confirmed fraud cases. No conclusions should be drawn about any specific provider without further investigation by qualified professionals.
