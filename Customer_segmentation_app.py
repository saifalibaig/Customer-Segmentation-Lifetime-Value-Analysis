
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import zipfile

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    zip_path = "online_retail_II.zip"  # ZIP file in repo root
    csv_filename = "online_retail_II.csv"

    # Unzip if not already unzipped
    if not os.path.exists(csv_filename):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")  # extract to repo root

    df = pd.read_csv(csv_filename)
    df = df.dropna(subset=["Customer ID", "Description"])
    df = df[df.Quantity > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df

df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
min_date = df["InvoiceDate"].min()
max_date = df["InvoiceDate"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["KMeans", "DBSCAN", "Hierarchical"]
)

df_filtered = df[
    (df["InvoiceDate"] >= pd.to_datetime(date_range[0])) &
    (df["InvoiceDate"] <= pd.to_datetime(date_range[1]))
]

# -----------------------------
# Build RFM Table
# -----------------------------
snapshot_date = df_filtered["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df_filtered.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# -----------------------------
# CLV Calculation (Simple)
# -----------------------------
rfm["CLV"] = rfm["Monetary"] * rfm["Frequency"]  # Basic CLV estimate

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# -----------------------------
# Clustering
# -----------------------------
if clustering_method == "KMeans":
    model = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = model.fit_predict(rfm_scaled)
    method_desc = "KMeans clusters customers by minimizing variance within clusters."

elif clustering_method == "DBSCAN":
    model = DBSCAN(eps=1.5, min_samples=5)
    rfm["Cluster"] = model.fit_predict(rfm_scaled)
    method_desc = "DBSCAN groups customers based on density and identifies outliers."

elif clustering_method == "Hierarchical":
    model = AgglomerativeClustering(n_clusters=4)
    rfm["Cluster"] = model.fit_predict(rfm_scaled)
    method_desc = "Hierarchical clustering groups customers into a tree-like structure."

# -----------------------------
# Map clusters to meaningful labels
# -----------------------------
cluster_labels = {
    0: "High Value Customers",
    1: "Medium Value Customers",
    2: "Low Value Customers",
    3: "Potential Value Customers"
}
rfm["ClusterLabel"] = rfm["Cluster"].map(cluster_labels)

# -----------------------------
# PCA for Visualization
# -----------------------------
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm["PCA1"] = rfm_pca[:, 0]
rfm["PCA2"] = rfm_pca[:, 1]

# -----------------------------
# Cluster Colors for Legend
# -----------------------------
cluster_colors = {
    "High Value Customers": "gold",
    "Medium Value Customers": "blue",
    "Low Value Customers": "red",
    "Potential Value Customers": "green"
}

# -----------------------------
# Dashboard Layout
# -----------------------------
st.title("📊 Customer Segmentation Dashboard with CLV")
st.markdown(f'''
This dashboard visualizes customer segments based on the **Online Retail II dataset**.
It uses **RFM analysis** + **{clustering_method} clustering** to segment customers.
''')

st.sidebar.markdown(f"**Clustering Method Info:** {method_desc}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segmentation", "Insights", "Documentation"])

# ===== Tab 1: Overview =====
with tab1:
    st.header("📌 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Total Sales", f"${df_filtered['TotalPrice'].sum():,.0f}")
    col2.metric("🏆 Total Customers", f"{df_filtered['Customer ID'].nunique():,}")
    col3.metric("🛒 Avg Order Value", f"${df_filtered['TotalPrice'].mean():,.2f}")
    col4.metric("📅 Time Range", f"{min(date_range)} → {max(date_range)}")
    col5.metric("📈 Avg CLV", f"${rfm['CLV'].mean():,.2f}")

    st.markdown("---")
    st.header("Sales Trend Over Time")
    sales_trend = df_filtered.groupby(df_filtered["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum().reset_index()
    sales_trend["InvoiceDate"] = sales_trend["InvoiceDate"].dt.to_timestamp()
    fig_sales = px.line(sales_trend, x="InvoiceDate", y="TotalPrice", title="Monthly Sales Trend")
    st.plotly_chart(fig_sales, use_container_width=True)
    st.markdown("This chart shows the trend of sales over the selected time range.")

    st.header("CLV Distribution")
    fig_clv = px.histogram(rfm, x="CLV", nbins=50, title="Customer Lifetime Value (CLV) Distribution")
    st.plotly_chart(fig_clv, use_container_width=True)
    st.markdown("This histogram shows how customer lifetime value (CLV) is distributed across customers.")

# ===== Tab 2: Segmentation =====
with tab2:
    st.header("🎯 Customer Segmentation")
    st.markdown("Segments are generated using RFM analysis and your selected clustering method.")

    fig_clusters = px.scatter(
        rfm, x="PCA1", y="PCA2",
        color="ClusterLabel",
        color_discrete_map=cluster_colors,
        size="Monetary",
        title="Customer Segmentation (PCA Projection)",
        labels={"PCA1": "PCA Component 1", "PCA2": "PCA Component 2"}
    )
    st.plotly_chart(fig_clusters, use_container_width=True)
    st.markdown("This plot shows customers in a 2D PCA space, colored by cluster labels.")

    st.markdown("### Cluster Legend")
    for label, color in cluster_colors.items():
        st.markdown(f"<span style='color:{color}'>■</span> {label}", unsafe_allow_html=True)

    st.markdown("### Cluster Summary Table")
    cluster_summary = rfm.groupby("ClusterLabel")[["Recency", "Frequency", "Monetary", "CLV"]].mean().reset_index()
    cluster_summary["Customer Count"] = rfm["ClusterLabel"].value_counts().sort_index().values
    st.dataframe(cluster_summary)

    st.download_button("📥 Download Cluster Data", rfm.to_csv(index=False), "clusters.csv")

    st.header("Treemap: Sales by Cluster")
    cluster_sales = df_filtered.merge(rfm[["CustomerID", "ClusterLabel"]], left_on="Customer ID", right_on="CustomerID")
    sales_cluster = cluster_sales.groupby("ClusterLabel")["TotalPrice"].sum().reset_index()
    fig_treemap = px.treemap(sales_cluster, path=["ClusterLabel"], values="TotalPrice", title="Sales Distribution by Cluster")
    st.plotly_chart(fig_treemap, use_container_width=True)
    st.markdown("The treemap shows the sales share for each cluster.")

    st.header("Heatmap: RFM Features by Cluster")
    heatmap_data = rfm.groupby("ClusterLabel")[["Recency", "Frequency", "Monetary", "CLV"]].mean()
    fig_heatmap = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="Viridis", title="RFM Features Heatmap by Cluster")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("The heatmap shows average RFM values for each cluster.")

    st.header("📄 Customers in Selected Cluster")
    selected_cluster = st.selectbox("Select Cluster", rfm["ClusterLabel"].unique())
    st.dataframe(rfm[rfm["ClusterLabel"] == selected_cluster])

# ===== Tab 3: Insights =====
with tab3:
    st.header("💡 Business Insights")
    st.markdown('''
    - **High Value Customers** → Reward loyalty with premium offers & personalized marketing.
    - **Medium Value Customers** → Upsell & cross-sell products to increase spend.
    - **Low Value Customers** → Re-engage with targeted campaigns & discounts.
    - **Potential Value Customers** → Convert with attractive onboarding promotions.
    ''')

# ===== Tab 4: Documentation =====
with tab4:
    st.header("📚 Documentation")
    st.markdown('''
    **RFM Analysis**:
    - **Recency**: How recently a customer made a purchase.
    - **Frequency**: How often they purchase.
    - **Monetary**: How much they spend.
    - **CLV**: Customer Lifetime Value — an estimate of a customer’s total value to a business over time.
    
    **Clustering**:
    - Groups customers with similar purchase behavior.
    - Methods include KMeans, DBSCAN, and Hierarchical Clustering.
    
    **Business Insights**:
    - Allows targeted marketing strategies based on cluster analysis.
    ''')
