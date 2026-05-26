# **Credit Card Fraud Detection System**
*Exploratory Data Analysis (EDA), Linear Regression & Interactive Dashboard using Kaggle Dataset*

---

## **Overview**

Manual review of credit card transactions is slow, error-prone, and completely impractical at scale. Financial institutions and payment processors handle millions of transactions daily — making it impossible to flag fraudulent activity without automated, data-driven systems.

This project focuses on building a **Credit Card Fraud Detection System** that performs **Exploratory Data Analysis (EDA)**, applies **Linear Regression**, and delivers an **Interactive Dashboard** on a real-world Kaggle dataset to understand transaction behavior and identify patterns associated with fraud.

By analyzing historical transaction data, the system detects anomalies, compares normal vs fraudulent transaction profiles, and visualizes relationships between key features — helping financial institutions respond faster and smarter to fraudulent activity.

The workflow includes:

- Data Collection: Loading the Kaggle Credit Card Fraud dataset
- Data Cleaning: Handling missing values and normalizing features
- Statistical Analysis: Understanding transaction amount and time distribution
- Fraud Behavior Analysis: Comparing normal vs fraudulent transaction patterns
- Group-Based Analysis: Examining fraud rates across time and amount ranges
- Visualization: Interactive charts, box plots, and correlation heatmaps
- Predictive Modeling: Linear Regression to estimate fraud likelihood
- Model Evaluation: R² score, MSE, RMSE, and residual analysis
- Dashboard: Interactive Plotly Dash application with filters
- Business Insights: Supporting faster and more accurate fraud prevention

---

## **Dataset**

**Source:** Kaggle
**Dataset Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
**Dataset Name:** Credit Card Fraud Detection Dataset

**Description:**
The dataset contains real-world anonymized credit card transactions made by European cardholders in September 2013. It includes 284,807 transactions over two days, of which 492 are fraudulent — making it a highly imbalanced dataset.

### **Selected Columns**

| Column Name | Description |
|---|---|
| `Time` | Seconds elapsed between this transaction and the first transaction |
| `Amount` | Transaction amount in euros |
| `V1 – V28` | Anonymized PCA-transformed features (original features confidential) |
| `Class` | Target column (0 = Normal, 1 = Fraud) |

### **Removed Columns**

- Duplicate transaction records
- Unnecessary identifiers
- Highly sparse columns
- Irrelevant metadata columns

Removed to simplify EDA and focus only on meaningful transaction features.

---

## **Objectives**

1. **Data Collection**
   - Load the Credit Card Fraud dataset from Kaggle
   - Select all relevant columns for analysis

2. **Data Cleaning & Preprocessing**
   - Check and handle missing values
   - Remove duplicate records
   - Normalize the `Amount` column using `StandardScaler`
   - Derive `Hour` column from `Time` for time-of-day analysis

3. **Descriptive Statistics**
   - Average transaction amount
   - Total fraudulent vs normal transaction counts
   - Overall fraud percentage
   - Class-wise descriptive statistics for Amount and Time

4. **Fraud Behavior Analysis**
   - Compare Normal vs Fraud transaction counts
   - Analyze average transaction amount differences across classes

5. **Group-Based Analysis**
   - Fraud rate by transaction amount range
   - Fraud rate by hour of day

6. **Relationship Analysis**
   - Time vs Amount scatter colored by class
   - Amount distribution overlay by fraud status
   - Correlation between key PCA features (V1–V28) and Class

7. **Data Visualization**
   - Bar charts → Fraud distribution and amount comparisons
   - Histograms → Transaction amount and PCA feature distributions
   - Scatter plots → Time vs Amount by transaction class
   - Box plots → Amount spread and outlier detection by class
   - Heatmaps → Correlation matrix for key features vs Class

8. **Risk Analysis**
   - Classify transactions into Low, Medium, and High Risk based on Amount
   - Calculate fraud rate per risk band to validate classification

9. **Predictive Modeling (Linear Regression)**
   - Independent variables: `Time`, `Amount`
   - Dependent variable: `Class` (fraud probability proxy)
   - Train/test split and model training
   - Predict fraud likelihood scores

10. **Model Evaluation**
    - Calculate R² Score
    - Calculate MSE and RMSE
    - Analyze residuals via scatter plot

11. **Dashboard Creation (Main Focus)**
    - Build interactive dashboard using Plotly Dash
    - Add user controls: Risk Level dropdown, Amount Range slider
    - Display four dynamic charts that update based on filters

12. **Insights & Conclusion**
    - Identify key fraud-driving factors
    - Understand transaction behavior patterns
    - Support better fraud prevention and risk management strategies

---

## **Project Highlights**

### **1. Data Collection & Cleaning**

- Dataset loaded from Kaggle (real European cardholder transactions, Sept 2013)
- `Amount` column normalized using `StandardScaler` for fair model input
- `Hour` column derived from `Time` (seconds elapsed) for time-of-day analysis
- Missing values filled with column means; duplicate records dropped

This step ensures data reliability and accuracy for all downstream analysis.

---

### **2. Exploratory Data Analysis (EDA)**

- Average transaction amount, fraud count, and fraud percentage calculated
- Normal vs fraudulent transactions compared across amount and time dimensions
- Fraud rates analyzed by amount range and hour of day
- Correlations between PCA features (V1–V28), Time, Amount, and Class explored

EDA helps uncover which transaction attributes most strongly indicate fraudulent behavior.

---

### **3. Visualization**

The project includes:

- **Bar Charts** – Normal vs Fraud counts and average amount comparisons
<img width="1365" height="559" alt="image" src="https://github.com/user-attachments/assets/75600dfa-c9b6-4b18-8aec-0b7f1b8c1aec" />
- **Line Charts** – Fraud rate across hours of day
<img width="1360" height="557" alt="image" src="https://github.com/user-attachments/assets/2d40961d-3ac0-4886-b5ba-862ee38e703a" />
- **Histograms** – Amount and top PCA feature distributions by class
<img width="1367" height="559" alt="image" src="https://github.com/user-attachments/assets/cd30fdad-dac3-43a7-b3d6-d453b02d61fb" />
- **Scatter Plots** – Time vs Amount colored by fraud status
<img width="1360" height="557" alt="image" src="https://github.com/user-attachments/assets/69d0e322-30f8-41bc-94b2-d70577ffda96" />
- **Box Plots** – Transaction amount spread and outlier detection by class
- <img width="1365" height="559" alt="image" src="https://github.com/user-attachments/assets/ab4e9a82-dd5b-407c-9fe2-e6655878f403" />
- **Heatmaps** – Correlation matrix across key PCA features and Class
<img width="995" height="866" alt="image" src="https://github.com/user-attachments/assets/fc0173e6-3aa5-400d-bcd7-7e8c46819d28" />

Visualizations make complex, high-dimensional transaction data easy to interpret and communicate.

---

### **4. Risk Classification**

Transactions are classified into three risk bands based on transaction amount:

| Risk Level | Transaction Amount |
|---|---|
| 🟢 Low Risk | Below $50 |
| 🟡 Medium Risk | $50 – $500 |
| 🔴 High Risk | Above $500 |

Fraud rate is then calculated per risk band to validate and interpret the classification.

---

### **5. Predictive Modeling (Linear Regression)**

A simple **Linear Regression** model is built to predict fraud likelihood based on:

- Transaction Time
- Transaction Amount

The model is trained on 80% of the data and evaluated on the remaining 20% using R², MSE, and RMSE. A residual scatter plot is generated to assess prediction quality.

> **Note:** Linear Regression serves as a transparent baseline model. For production fraud detection, ensemble classifiers such as Random Forest or XGBoost with SMOTE oversampling are recommended due to severe class imbalance.

---

### **6. Interactive Dashboard**

Built using **Plotly Dash**, the dashboard provides:

- Risk Level dropdown for category-based filtering
- Amount Range slider for fine-grained transaction filtering
- **Scatter Chart** – Time vs Amount (filtered view, colored by class)
- **Bar Chart** – Fraud count by hour of day
- **Pie Chart** – Normal vs Fraud distribution
- **Histogram** – Transaction amount distribution by class

The dashboard transforms static analysis into an interactive fraud exploration tool.

---

## **Tools and Technologies**

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Plotly Dash
- Scikit-learn
- Jupyter Notebook
- Google Colab
- Kaggle Dataset
- CSV File Storage

---

## **Results**

### **Key Findings**

- **Fraud Rate:** Only ~0.17% of all transactions are fraudulent, confirming severe class imbalance.
- **Amount Patterns:** Fraudulent transactions tend to cluster at lower amounts, suggesting deliberate concealment behavior.
- **Time Patterns:** Fraud rate spikes at specific hours of the day, indicating time-sensitive attack windows.
- **PCA Features:** V14, V12, V10, and V4 show the strongest negative correlation with the fraud class.
- **Risk Bands:** High-value transactions (above $500) carry a disproportionately elevated fraud risk.
- **Amount vs Class:** Normal transactions have a higher average amount than fraudulent ones, contrary to intuition, as fraudsters prefer smaller, less detectable charges.

---

### **Interpretation**

- **Fraud Prevention:** Financial institutions can use time-of-day and amount thresholds to trigger additional authentication steps.
- **Feature Intelligence:** PCA features V14 and V12 are the most actionable signals for fraud scoring models.
- **Risk Band Policies:** Tiered risk classification supports differential transaction monitoring and automated hold policies.
- **Class Imbalance Awareness:** The 0.17% fraud rate emphasizes the need for precision-recall focused evaluation over simple accuracy.
- **Baseline Modeling:** Even linear regression reveals which transaction features correlate most with fraud, guiding feature selection for advanced models.

The structured EDA and dashboard approach improves fraud visibility and supports faster, more accurate detection decisions.

---

## **Conclusion**

The **Credit Card Fraud Detection System** successfully analyzes real-world anonymized transaction data and provides actionable insights into fraudulent behavior patterns.

Through structured **EDA, risk classification, predictive modeling, and interactive visualization**, the system identifies high-risk transactions, key fraud-driving features, and temporal patterns — enabling financial institutions to build smarter, more responsive fraud detection pipelines.

This project demonstrates how **data analytics and interactive dashboards can transform raw transaction data into practical fraud intelligence**.

---

## **Author**

**Shree Pranava Ganesh**
Student at Kamaraj College
Thoothukudi, Tamil Nadu
