import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Student K-Means Clustering",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Student K-Means Clustering App")

st.write(
    "Upload student data in CSV format. "
    "The app will cluster students based on selected numeric variables."
)

# ---------------------------------------------------
# Detect numeric columns
# ---------------------------------------------------
def detect_numeric_columns(df):
    numeric_cols = []

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().mean() >= 0.8:
            numeric_cols.append(col)

    return numeric_cols


# ---------------------------------------------------
# Prepare data
# ---------------------------------------------------
def prepare_data(df, selected_cols):

    X = df[selected_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    return X


# ---------------------------------------------------
# PCA Cluster Plot
# ---------------------------------------------------
def plot_clusters(X_scaled, labels):

    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        components[:, 0],
        components[:, 1],
        c=labels,
        cmap="tab10",
        s=80,
        alpha=0.8
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.set_title("Student Clusters (PCA Visualization)")

    ax.grid(alpha=0.3)

    # Legend
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )

    ax.add_artist(legend)

    return fig


# ---------------------------------------------------
# Upload CSV
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------------------------------------------
# Main App
# ---------------------------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(df.head())

    numeric_cols = detect_numeric_columns(df)

    if len(numeric_cols) == 0:
        st.error("No numeric columns detected.")
        st.stop()

    # Remove label columns if present
    default_cols = [
        col for col in numeric_cols
        if col.lower() not in ["cluster", "true_group"]
    ]

    selected_cols = st.multiselect(
        "Select variables for clustering",
        numeric_cols,
        default=default_cols
    )

    k = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    # ---------------------------------------------------
    # Run Clustering
    # ---------------------------------------------------
    if st.button("Run K-Means Clustering"):

        if len(selected_cols) == 0:
            st.warning("Please select at least one variable.")
            st.stop()

        # Prepare Data
        X = prepare_data(df, selected_cols)

        # Standardize
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # KMeans Model
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        # Add cluster labels
        df["cluster"] = labels

        # ---------------------------------------------------
        # Results
        # ---------------------------------------------------
        st.subheader("Clustered Student Data")

        st.dataframe(df)

        # Silhouette Score
        sil = silhouette_score(X_scaled, labels)

        st.metric(
            label="Silhouette Score",
            value=round(sil, 3)
        )

        # ---------------------------------------------------
        # Cluster Profiles
        # ---------------------------------------------------
        st.subheader("Cluster Profiles")

        cluster_summary = (
            df.groupby("cluster")[selected_cols]
            .mean()
            .round(2)
        )

        st.dataframe(cluster_summary)

        st.write(
            """
            Interpretation Guide:
            - Higher averages → Higher performing cluster
            - Lower averages → Lower performing cluster
            """
        )

        # ---------------------------------------------------
        # PCA Plot
        # ---------------------------------------------------
        st.subheader("Cluster Visualization")

        fig = plot_clusters(X_scaled, labels)

        st.pyplot(fig)

        # ---------------------------------------------------
        # Download Results
        # ---------------------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Clustered CSV",
            data=csv,
            file_name="clustered_students.csv",
            mime="text/csv"
        )
