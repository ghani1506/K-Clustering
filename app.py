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
    ax.set_title("Student Clusters Using PCA")
    ax.grid(alpha=0.3)

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )

    ax.add_artist(legend)

    return fig


uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    numeric_cols = detect_numeric_columns(df)

    if len(numeric_cols) == 0:
        st.error("No numeric columns detected.")
        st.stop()

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

    if st.button("Run K-Means Clustering"):

        if len(selected_cols) == 0:
            st.warning("Please select at least one variable.")
            st.stop()

        X = prepare_data(df, selected_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        df["cluster"] = labels

        # ---------------------------------------------------
        # Cluster Profiles
        # ---------------------------------------------------
        cluster_summary = (
            df.groupby("cluster")[selected_cols]
            .mean()
            .round(2)
        )

        cluster_summary["Overall_Mean"] = (
            cluster_summary[selected_cols]
            .mean(axis=1)
            .round(2)
        )

        cluster_summary = cluster_summary.sort_values(
            by="Overall_Mean"
        )

        ordered_clusters = cluster_summary.index.tolist()

        performance_labels = {}

        if len(ordered_clusters) == 2:
            performance_labels[ordered_clusters[0]] = "Low Performers"
            performance_labels[ordered_clusters[1]] = "High Performers"

        elif len(ordered_clusters) == 3:
            performance_labels[ordered_clusters[0]] = "Low Performers"
            performance_labels[ordered_clusters[1]] = "Average Performers"
            performance_labels[ordered_clusters[2]] = "High Performers"

        else:
            performance_labels[ordered_clusters[0]] = "Lowest Performers"
            performance_labels[ordered_clusters[-1]] = "Highest Performers"

            for middle_cluster in ordered_clusters[1:-1]:
                performance_labels[middle_cluster] = "Middle Performers"

        cluster_summary["Performance_Level"] = (
            cluster_summary.index.map(performance_labels)
        )

        df["Performance_Level"] = df["cluster"].map(performance_labels)

        # ---------------------------------------------------
        # Results
        # ---------------------------------------------------
        st.subheader("Clustered Student Data")
        st.dataframe(df)

        sil = silhouette_score(X_scaled, labels)

        st.metric(
            label="Silhouette Score",
            value=round(sil, 3)
        )

        st.subheader("Cluster Profiles: Lowest to Highest Performers")
        st.dataframe(cluster_summary)

        st.write(
            """
            **How to read this:**

            The app calculates the average score for each cluster.

            The cluster with the **lowest Overall_Mean** is labelled as the weakest group.

            The cluster with the **highest Overall_Mean** is labelled as the strongest group.
            """
        )

        st.subheader("Cluster Visualization")

        fig = plot_clusters(X_scaled, labels)
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Clustered CSV",
            data=csv,
            file_name="clustered_students.csv",
            mime="text/csv"
        )
