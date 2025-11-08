import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

# ----------------------------
# Load saved objects
# ----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca_cluster.pkl", "rb") as f:
    pca_cluster = pickle.load(f)

with open("pca_2d.pkl", "rb") as f:
    pca_2d = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    cps_oe = pickle.load(f)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Customer Segmentation App (KMeans)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    # ----------------------------
    # Preprocessing
    # ----------------------------
    # Age
    if "Year_Birth" in df.columns:
        df["Age"] = 2025 - df["Year_Birth"]

    # Marital_Status
    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace(
            {"Together":"Married","Widow":"Divorced","Alone":"Single","YOLO":"Absurd"}
        )

    # Education
    if "Education" in df.columns:
        df["Education"] = df["Education"].replace({"2n Cycle":"Master"})
        df["Education"] = cps_oe.transform(df[["Education"]])

    # Map Marital_Status
    marital_mapping = {"Absurd":0, "Single":1, "Married":2, "Divorced":3}
    df["Marital_Status"] = df["Marital_Status"].map(marital_mapping)

    # Drop unnecessary columns
    drop_cols = ["ID","Year_Birth","Dt_Customer","Z_Revenue"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Fill missing values for Income
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].mean())

    # Standardize
    std_col = ["Income","Kidhome","Teenhome","Recency","MntWines","MntFruits",
               "MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds",
               "NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",
               "NumStorePurchases","NumWebVisitsMonth","AcceptedCmp3","AcceptedCmp4",
               "AcceptedCmp5","AcceptedCmp1","AcceptedCmp2","Complain","Z_CostContact",
               "Response","Age"]
    df[std_col] = scaler.transform(df[std_col])

    # ----------------------------
    # PCA and KMeans Prediction
    # ----------------------------
    df_reduced = pca_cluster.transform(df[std_col])
    df_2d = pca_2d.transform(df[std_col])
    df["Cluster"] = kmeans.predict(df_reduced)

    st.write("Cluster Assignments:")
    st.dataframe(df[["Cluster"] + std_col])

    # ----------------------------
    # Visualization
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(df_2d[:,0], df_2d[:,1], c=df["Cluster"], cmap='viridis', alpha=0.6)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("Customer Clusters (2D PCA)")
    plt.colorbar(scatter, label="Cluster")
    st.pyplot(fig)
