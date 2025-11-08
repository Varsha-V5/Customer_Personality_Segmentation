# Customer_Personality_Segmentation
# Customer Segmentation Clustering

## Project Overview

This project demonstrates **customer segmentation** using **unsupervised machine learning** techniques. The goal is to group customers into meaningful segments based on their demographic and transactional behavior, which can then be used for **targeted marketing, personalized offers, and business insights**.

The project uses:  
- **KMeans clustering** (primary, predictive model)  
- **Agglomerative Clustering and DBSCAN** (exploratory clustering)  
- **PCA** for dimensionality reduction and visualization  
- **Streamlit app** for interactive deployment

---

## Dataset

The dataset `customer_segmentation.csv` includes customer demographic and purchase information. Key features:

| Feature | Description |
|---------|-------------|
| `ID` | Customer identifier |
| `Year_Birth` | Year of birth |
| `Education` | Education level |
| `Marital_Status` | Marital status |
| `Income` | Annual income |
| `Kidhome`, `Teenhome` | Number of kids/teens in the household |
| `Recency` | Days since last purchase |
| `MntWines`, `MntFruits`, ... | Amount spent on different product categories |
| `NumWebPurchases`, `NumCatalogPurchases`, ... | Number of purchases via different channels |
| `AcceptedCmp1`, ... | Campaign responses |
| `Response` | Whether the customer responded to the last campaign |

---

## Preprocessing

1. **Handle missing values**: `Income` imputed with mean.  
2. **Feature engineering**: Added `Age` = 2025 - `Year_Birth`.  
3. **Encode categorical features**:  
   - `Education` → ordinal values (`Basic` < `Graduation` < `Master` < `PhD`)  
   - `Marital_Status` → mapped to integers (`Absurd=0, Single=1, Married=2, Divorced=3`)  
4. **Standardization**: Features scaled using `StandardScaler` to have mean 0 and variance 1.  
5. **Dimensionality reduction**: PCA applied for both clustering (multi-component) and 2D visualization.

---

## Clustering Models

1. **KMeans Clustering**  
   - Used for **predictive clustering** (can assign new/test data to clusters).  
   - `n_clusters=2` (based on silhouette analysis).  
   - Silhouette Score (Train/Test) computed to measure cluster quality.

2. **Agglomerative Clustering** (exploratory)  
   - Hierarchical clustering using `ward` linkage.  
   - Only used on training data (cannot predict new data).  

3. **DBSCAN** (exploratory)  
   - Density-based clustering to detect **outliers** and dense clusters.  
   - Noise points (`-1`) ignored when computing silhouette score.

---

## Visualization

- PCA-reduced 2D scatter plots to visualize cluster separation for both **train** and **test** datasets.  
- Each cluster colored differently for easy interpretation.  

---

## Streamlit Deployment

The Streamlit app allows users to:  

1. Upload a **new customer CSV file**.  
2. Preprocess it automatically (encoding, scaling, PCA).  
3. Predict **KMeans cluster labels**.  
4. Display **2D PCA scatter plot** of clusters.  
5. Show **cluster summaries** (mean values for each feature).  

### How to Run

1. Install dependencies:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
