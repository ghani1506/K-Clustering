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

    ax.set_xlabel("Performance Pattern 1")
    ax.set_ylabel("Performance Pattern 2")
    ax.set_title("Student Clusters Using PCA")
    ax.grid(alpha=0.3)

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )

    ax.add_artist(legend)

    return fig


def create_performance_labels(cluster_summary):
    ordered_clusters = cluster_summary.sort_values(
        by="Overall_Mean"
    ).index.tolist()

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

    return performance_labels


def generate_cluster_interpretation(cluster_summary, selected_cols):
    interpretations = []

    overall_average = cluster_summary["Overall_Mean"].mean()

    for cluster_id, row in cluster_summary.iterrows():
        performance_level = row["Performance_Level"]

        strongest_subject = row[selected_cols].idxmax()
        weakest_subject = row[selected_cols].idxmin()

        strongest_score = row[strongest_subject]
        weakest_score = row[weakest_subject]
        overall_mean = row["Overall_Mean"]

        if overall_mean >= overall_average:
            general_statement = "This cluster performs above the overall cluster average."
        else:
            general_statement = "This cluster performs below the overall cluster average."

        interpretation = f"""
### Cluster {cluster_id}: {performance_level}

- **Overall Mean:** {overall_mean}
- **Strongest Area:** {strongest_subject} ({strongest_score})
- **Weakest Area:** {weakest_subject} ({weakest_score})
- **Interpretation:** {general_statement}
- **Suggested Action:** Provide enrichment if this is a high-performing group, or targeted support if this group has weaker subject areas.
"""
        interpretations.append(interpretation)

    return "\n".join(interpretations)


def generate_conclusion(cluster_summary):
    lowest_cluster = cluster_summary["Overall_Mean"].idxmin()
    highest_cluster = cluster_summary["Overall_Mean"].idxmax()

    lowest_label = cluster_summary.loc[lowest_cluster, "Performance_Level"]
    highest_label = cluster_summary.loc[highest_cluster, "Performance_Level"]

    conclusion = f"""
### Overall Conclusion

The K-Means clustering analysis identified **{len(cluster_summary)} distinct student performance groups**.

The strongest group is **Cluster {highest_cluster} ({highest_label})**, while the weakest group is **Cluster {lowest_cluster} ({lowest_label})**.

This suggests that student performance is not uniform. Different clusters show different learning profiles, strengths, and weaknesses. Therefore, teaching support should be more targeted rather than applying the same intervention to all students.

For example:

- Higher-performing clusters may benefit from enrichment and challenge tasks.
- Middle-performing clusters may benefit from guided practice and feedback.
- Lower-performing clusters may require focused intervention, remediation, and closer monitoring.
"""

    return conclusion


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

        performance_labels = create_performance_labels(cluster_summary)

        cluster_summary["Performance_Level"] = (
            cluster_summary.index.map(performance_labels)
        )

        df["Performance_Level"] = df["cluster"].map(performance_labels)

        st.subheader("Clustered Student Data")
        st.dataframe(df)

        sil = silhouette_score(X_scaled, labels)

        st.metric(
            label="Silhouette Score",
            value=round(sil, 3)
        )

        st.subheader("Cluster Profiles: Lowest to Highest Performers")

        cluster_summary = cluster_summary.sort_values(
            by="Overall_Mean"
        )

        st.dataframe(cluster_summary)

        st.subheader("Cluster Visualization")

        fig = plot_clusters(X_scaled, labels)
        st.pyplot(fig)

        st.subheader("Automatic Interpretation")

        interpretation_text = generate_cluster_interpretation(
            cluster_summary,
            selected_cols
        )

        st.markdown(interpretation_text)

        st.subheader("Automatic Conclusion")

        conclusion_text = generate_conclusion(cluster_summary)

        st.markdown(conclusion_text)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Clustered CSV",
            data=csv,
            file_name="clustered_students.csv",
            mime="text/csv"
        )
