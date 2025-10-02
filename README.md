# 📊 Customer Segmentation & CLV Analysis

An interactive Streamlit dashboard that analyzes customer purchase behavior using the **Online Retail II** dataset.  
The app performs **RFM (Recency, Frequency, Monetary)** analysis, estimates **Customer Lifetime Value (CLV)**, and segments customers using clustering (KMeans, DBSCAN, Hierarchical). This helps businesses identify high-value customers, re-engage low-value customers, and design targeted marketing strategies.

🔗 **Live Demo:** https://customer-segmentation-lifetime-value-analysis.streamlit.app/

---

## ✨ Key Highlights
- RFM-based segmentation and CLV estimation  
- Multiple clustering options: **KMeans**, **DBSCAN**, **Hierarchical**  
- PCA visualization for cluster interpretation  
- Visuals include histograms, treemaps, heatmaps, and trend charts  
- Downloadable clustered customer dataset for further analysis  
- Interactive date-range filters and clustering method selection

---

## 🚀 Features
- **RFM Table**: Constructs Recency, Frequency, and Monetary metrics per customer.  
- **CLV Estimation**: Basic CLV computed using Frequency × Monetary (extendable to predictive CLV).  
- **Clustering**: Choose clustering method via sidebar and map clusters to business-friendly labels.  
- **Visualization**: PCA scatter, cluster treemap, CLV distribution, RFM heatmap, and monthly sales trend.  
- **Export**: Download clusters as CSV for offline analysis or campaign targeting.  
- **Documentation Tab**: In-app explanations of RFM, CLV, and clustering.

---

## 📂 Project Structure
~~~bash
.
├── customer_segmentation_app.py   # Streamlit app source code
├── online_retail_II.zip           # Dataset (compressed CSV)
├── online_retail_II.csv           # Extracted dataset (or extracted at runtime)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
~~~

---

## 📊 RFM & CLV Overview
- **Recency (R):** Days since the customer's last purchase (lower is better).  
- **Frequency (F):** Number of unique purchases/invoices (higher is better).  
- **Monetary (M):** Total spending amount.  
- **CLV (basic):** `CLV = Frequency × Monetary` — a simple estimate to rank customer value quickly. (Replaceable with predictive CLV models.)

---

## 🧠 Clustering Methods
- **KMeans:** Partitions customers into `k` clusters minimizing within-cluster variance.  
- **DBSCAN:** Density-based clustering; useful for discovering core clusters and identifying outliers.  
- **Hierarchical (Agglomerative):** Builds a nested cluster tree; useful when cluster numbers are uncertain.

Clusters are mapped to readable labels such as:
- High Value Customers  
- Medium Value Customers  
- Low Value Customers  
- Potential Value Customers

---

## 🎯 Business Use Cases
- **Targeted Marketing Campaigns:** Offer premium deals to high-value customers.  
- **Customer Retention:** Identify and re-engage at-risk or low-value customers.  
- **Upsell/Cross-sell Strategy:** Focus on medium-value segments for conversion.  
- **Segmentation-driven Reporting:** Use exported clusters for CRM and campaign automation.

---

## 📈 How to Run Locally
1. Clone the repo:
~~~bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
~~~
2. Install dependencies:
~~~bash
pip install -r requirements.txt
~~~
3. Run the Streamlit app:
~~~bash
streamlit run customer_segmentation_app.py
~~~

---

## ⚙️ Requirements
- Python 3.8+  
- Libraries used: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`

(Use `pip freeze > requirements.txt` in your environment to capture exact versions.)

---

## 📈 Future Enhancements
- Build **predictive CLV** using supervised models (XGBoost, LightGBM, or LSTM).  
- Add **cohort analysis** and churn prediction dashboards.  
- Enable **product/region-level drill-downs** and campaign ROI tracking.  
- Integrate **real-time data** sources (SQL, BigQuery) and authentication for enterprise use.  
- Provide **SHAP-based explainability** for CLV and segment assignment.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙋 Contact
Built by [Your Name].  
For questions or collaboration: `youremail@example.com` or visit my GitHub profile: `https://github.com/yourusername`

