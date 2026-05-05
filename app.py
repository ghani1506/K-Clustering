import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Student K-Means Clustering", page_icon="🎯", layout="wide")

st.title("🎯 Student K-Means Clustering App")
st.write("Upload student data in CSV format. The app will cluster students based on selected numeric variables.")

def detect_numeric_columns(df):
    numeric_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            numeric_cols.append(col)
    return numeric_cols

def prepare_data(df, selected_cols):
    X = df[selected_cols].apply(pd.to_numeric, errors="coerce")
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    return X

def plot_clusters(X_scaled, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(components[:,0], components[:,1], c=labels, s=60, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters (PCA)")
    ax.grid(alpha=0.3)
    return fig

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    numeric_cols = detect_numeric_columns(df)

    selected_cols = st.multiselect("Select variables", numeric_cols, default=numeric_cols)
    k = st.slider("Clusters", 2, 10, 3)

    if st.button("Run"):
        X = prepare_data(df, selected_cols)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)

        df["cluster"] = labels
        st.dataframe(df)

        sil = silhouette_score(X_scaled, labels)
        st.write("Silhouette:", round(sil,3))

        fig = plot_clusters(X_scaled, labels)
        st.pyplot(fig)

        st.download_button("Download CSV", df.to_csv(index=False), "clusters.csv")
