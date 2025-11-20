import argparse
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def merge_dfs(summary_dir: str) -> pd.DataFrame:
    summary_files = pathlib.Path(summary_dir).glob("*.csv")
    df_list = [pd.read_csv(f) for f in summary_files]
    return pd.concat(df_list, ignore_index=True)


def get_embeddings(model, articles_concepts: dict):
    article_ids, article_vectors = [], []

    for art_id, concepts in articles_concepts.items():
        concept_embs = model.encode(concepts)
        article_vec = np.mean(concept_embs, axis=0)
        article_ids.append(art_id)
        article_vectors.append(article_vec)

    return article_ids, np.vstack(article_vectors)


def interpret_axes(concept_2d, unique_concepts, top=10):
    pc1 = concept_2d[:, 0]
    pc2 = concept_2d[:, 1]

    def top_terms(values):
        return {
            "pos": [unique_concepts[i] for i in np.argsort(values)[-top:]],
            "neg": [unique_concepts[i] for i in np.argsort(values)[:top]],
        }

    return {"PC1": top_terms(pc1), "PC2": top_terms(pc2)}


def build_axis_labels(axis_info, max_terms=5):
    def fmt(name, terms):
        return f"{name}: " + ", ".join(terms[:max_terms])

    return {
        "PC1_pos": fmt("PC1+", axis_info["PC1"]["pos"]),
        "PC1_neg": fmt("PC1−", axis_info["PC1"]["neg"]),
        "PC2_pos": fmt("PC2+", axis_info["PC2"]["pos"]),
        "PC2_neg": fmt("PC2−", axis_info["PC2"]["neg"]),
    }


# ---------- ELLIPSE UTILITY ----------
def compute_cluster_ellipse(points_2d: np.ndarray, n_std: float = 2.0, n_points: int = 100):
    """Return ellipse around a cluster using covariance."""
    if points_2d.shape[0] < 2:
        return None, None

    mean = points_2d.mean(axis=0)
    cov = np.cov(points_2d, rowvar=False)

    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    axes = n_std * np.sqrt(vals)
    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(t), np.sin(t)])
    ellipse = (vecs @ np.diag(axes) @ circle).T + mean

    return ellipse[:, 0], ellipse[:, 1]


# ---------- CLUSTER NAMING ----------
def compute_cluster_names(
    article_ids,
    article_vectors,
    articles_concepts,
    cluster_labels,
    unique_concepts,
    concept_embs,
    max_terms: int = 5,
):
    """
    Assign a semantic label to each cluster based on:
    - most frequent concepts in the cluster
    - concepts closest to cluster centroid in embedding space
    """
    cluster_names = {}
    for cl in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cl)[0]

        # Bag-of-concepts: frequency within cluster
        counter = Counter()
        for i in idx:
            aid = article_ids[i]
            counter.update(articles_concepts[aid])
        top_freq = [w for w, _ in counter.most_common(max_terms)]

        # Centroid in embedding space
        centroid = article_vectors[idx].mean(axis=0)
        sims = cosine_similarity(centroid.reshape(1, -1), concept_embs)[0]
        top_sim_idx = np.argsort(sims)[-max_terms * 2:]  # a bit more, then dedupe
        top_sim = [unique_concepts[j] for j in top_sim_idx[::-1]]

        # Combine (freq first, then similar) without duplicates
        combined = []
        for w in top_freq + top_sim:
            if w not in combined:
                combined.append(w)
            if len(combined) >= max_terms:
                break

        label = ", ".join(combined) if combined else "Miscellaneous"
        cluster_names[cl] = label

    return cluster_names


# ---------- PLOTTING ----------
def plotly_plot(
    article_ids,
    article_2d,
    concept_2d,
    concept_labels,
    cluster_labels,
    cluster_names,
    axis_labels,
    pca_var_ratio,
    output_file="plot.html",
):
    fig = go.Figure()

    x_min, x_max = article_2d[:, 0].min(), article_2d[:, 0].max()
    y_min, y_max = article_2d[:, 1].min(), article_2d[:, 1].max()

    pc1_var = pca_var_ratio[0] * 100
    pc2_var = pca_var_ratio[1] * 100

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # --- CLUSTER ELLIPSES (same colors as points) ---
    centroids = []
    centroid_texts = []
    for clust in unique_clusters:
        pts = article_2d[cluster_labels == clust]
        xs, ys = compute_cluster_ellipse(pts, n_std=2.0)
        if xs is None:
            continue

        # Color from Viridis colorscale
        color = px.colors.sample_colorscale(
            "Viridis", clust / (n_clusters - 1) if n_clusters > 1 else 0
        )[0]
        fillcolor = color.replace("rgb", "rgba").replace(")", ",0.15)")

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=2, color=color),
                fill="toself",
                fillcolor=fillcolor,
                hoverinfo="skip",
                opacity=0.3,
                showlegend=False,
            )
        )

        # Save centroid for labeling
        centroids.append(pts.mean(axis=0))
        centroid_texts.append(cluster_names.get(clust, f"Cluster {clust}"))

    # --- ARTICLES ---
    fig.add_trace(
        go.Scatter(
            x=article_2d[:, 0],
            y=article_2d[:, 1],
            mode="markers",
            marker=dict(size=10, color=cluster_labels, colorscale="Viridis"),
            hovertext=[
                f"{aid}<br>Cluster {clust}: {cluster_names.get(clust, '')}"
                for aid, clust in zip(article_ids, cluster_labels)
            ],
            hoverinfo="text",
            name="Articles",
        )
    )

    # --- CLUSTER LABEL TEXT AT CENTROIDS ---
    if centroids:
        centroids = np.array(centroids)
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="text",
                text=centroid_texts,
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --- IMPORTANT CONCEPTS ---
    fig.add_trace(
        go.Scatter(
            x=concept_2d[:, 0],
            y=concept_2d[:, 1],
            mode="markers",
            marker=dict(symbol="x", size=9),
            name="Axis concepts",
            hovertext=concept_labels,
            hoverinfo="text",
        )
    )

    # --- PCA AXIS LINES ---
    fig.add_shape(type="line", x0=x_min, y0=0, x1=x_max, y1=0,
                  line=dict(dash="dash", width=1, color="gray"))
    fig.add_shape(type="line", x0=0, y0=y_min, x1=0, y1=y_max,
                  line=dict(dash="dash", width=1, color="gray"))

    # --- AXIS LABELS ---
    fig.add_annotation(x=x_max, y=0, text=axis_labels["PC1_pos"],
                       xanchor="right", yanchor="bottom", showarrow=False)
    fig.add_annotation(x=x_min, y=0, text=axis_labels["PC1_neg"],
                       xanchor="left", yanchor="top", showarrow=False)
    fig.add_annotation(x=0, y=y_max, text=axis_labels["PC2_pos"],
                       xanchor="left", yanchor="bottom", showarrow=False)
    fig.add_annotation(x=0, y=y_min, text=axis_labels["PC2_neg"],
                       xanchor="left", yanchor="top", showarrow=False)

    # --- LAYOUT ---
    fig.update_layout(
        title="Articles in PCA Space (cluster ellipses + axis concepts)",
        xaxis_title=f"PC1 ({pc1_var:.1f} %)",
        yaxis_title=f"PC2 ({pc2_var:.1f} %)",
        width=1000,
        height=700,
    )

    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")


# ---------- MAIN PROCESS ----------
def plot_articles_and_concepts(
    article_ids,
    article_vectors,
    articles_concepts,
    model,
    n_clusters=4,
    top_axis_terms=10,
    output_html="articles_pca_plot.html",
):

    unique_concepts = sorted({c for lst in articles_concepts.values() for c in lst})
    concept_embs = model.encode(unique_concepts)

    # PCA
    pca = PCA(n_components=2)
    article_2d = pca.fit_transform(article_vectors)
    concept_2d = pca.transform(concept_embs)

    axis_info = interpret_axes(concept_2d, unique_concepts, top=top_axis_terms)
    axis_labels = build_axis_labels(axis_info)

    print("\nAxis interpretation:")
    for name, text in axis_labels.items():
        print(f"  {name}: {text}")

    # Filter important concepts
    axis_words = set(
        axis_info["PC1"]["pos"] + axis_info["PC1"]["neg"] +
        axis_info["PC2"]["pos"] + axis_info["PC2"]["neg"]
    )
    mask = [c in axis_words for c in unique_concepts]
    concept_2d_filtered = concept_2d[mask]
    concept_labels_filtered = [c for c in unique_concepts if c in axis_words]

    # Clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(article_vectors)

    # Cluster sizes
    print("\nArticles per cluster:")
    for cl in np.unique(cluster_labels):
        print(f"  Cluster {cl}: {(cluster_labels == cl).sum()} articles")

    # Cluster naming
    cluster_names = compute_cluster_names(
        article_ids,
        article_vectors,
        articles_concepts,
        cluster_labels,
        unique_concepts,
        concept_embs,
        max_terms=5,
    )

    print("\nCluster themes:")
    for cl in sorted(cluster_names.keys()):
        print(f"  Cluster {cl}: {cluster_names[cl]}")

    # Plot
    plotly_plot(
        article_ids,
        article_2d,
        concept_2d_filtered,
        concept_labels_filtered,
        cluster_labels,
        cluster_names,
        axis_labels,
        pca_var_ratio=pca.explained_variance_ratio_,
        output_file=output_html,
    )


def main(summary_dir: str, output_html: str):
    df = merge_dfs(summary_dir)

    articles_concepts = {
        row["Article"]: eval(row["Content"])
        for _, row in df.iterrows()
    }

    model = SentenceTransformer("all-MiniLM-L6-v2")
    article_ids, article_vectors = get_embeddings(model, articles_concepts)

    plot_articles_and_concepts(
        article_ids,
        article_vectors,
        articles_concepts,
        model,
        output_html=output_html,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument(
        "--output_html", default="articles_pca_plot.html"
    )
    args = parser.parse_args()

    main(args.summary_dir, args.output_html)
