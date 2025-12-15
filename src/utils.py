"""
Utility functions for data processing, clustering, and analysis.
"""

import ast
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# ---------- DATA PROCESSING ----------
def merge_dfs(summary_dir: str) -> pd.DataFrame:
    """Merge all CSV files from a directory into a single DataFrame."""
    summary_files = pathlib.Path(summary_dir).glob("*.csv")
    df_list = [pd.read_csv(f) for f in summary_files]
    return pd.concat(df_list, ignore_index=True)


def safe_parse_concepts(content_str):
    """Safely parse concept list from string, handling various formats."""
    if pd.isna(content_str):
        return []

    content_str = str(content_str).strip()

    # If it's empty, return empty list
    if not content_str:
        return []

    # Handle markdown code blocks (remove ```python and ``` markers)
    if content_str.startswith("```python") and content_str.endswith("```"):
        # Extract content between the markdown code block markers
        content_str = content_str[9:-3].strip()  # Remove ```python and ```
    elif content_str.startswith("```") and content_str.endswith("```"):
        # Handle generic code blocks
        content_str = content_str[3:-3].strip()  # Remove ``` and ```

    try:
        # Try to parse as literal (safer than eval)
        return ast.literal_eval(content_str)
    except (ValueError, SyntaxError):
        # If that fails, try to parse as comma-separated values
        try:
            # Remove quotes and brackets, split by comma
            cleaned = content_str.strip("[]\"'").replace("'", "").replace('"', "")
            return [item.strip() for item in cleaned.split(",") if item.strip()]
        except Exception:
            # Last resort: return as single item
            return [content_str]


def get_embeddings(model, articles_concepts: dict):
    """Generate embeddings for articles based on their concepts."""
    article_ids, article_vectors = [], []

    for art_id, concepts in articles_concepts.items():
        concept_embs = model.encode(concepts)
        article_vec = np.mean(concept_embs, axis=0)
        print(article_vec.shape)
        article_ids.append(art_id)
        article_vectors.append(article_vec)

    return article_ids, np.vstack(article_vectors)


# ---------- UMAP INTERPRETATION ----------
def interpret_axes(concept_2d, unique_concepts, top=10):
    """Interpret UMAP axes by finding concepts with highest/lowest loadings."""
    umap1 = concept_2d[:, 0]
    umap2 = concept_2d[:, 1]

    def top_terms(values):
        return {
            "pos": [unique_concepts[i] for i in np.argsort(values)[-top:]],
            "neg": [unique_concepts[i] for i in np.argsort(values)[:top]],
        }

    return {"UMAP1": top_terms(umap1), "UMAP2": top_terms(umap2)}


def build_axis_labels(axis_info, max_terms=5):
    """Build axis labels from UMAP interpretation."""

    def fmt(name, terms):
        return f"{name}: " + ", ".join(terms[:max_terms])

    return {
        "UMAP1_pos": fmt("UMAP1+", axis_info["UMAP1"]["pos"]),
        "UMAP1_neg": fmt("UMAP1−", axis_info["UMAP1"]["neg"]),
        "UMAP2_pos": fmt("UMAP2+", axis_info["UMAP2"]["pos"]),
        "UMAP2_neg": fmt("UMAP2−", axis_info["UMAP2"]["neg"]),
    }


# ---------- CLUSTER OPTIMIZATION ----------
def silhouette_method(
    article_vectors, k_range=None, output_file="silhouette_plot.html"
):
    """
    Use the silhouette method to find optimal number of clusters.

    Args:
        article_vectors: The embedding vectors for articles
        k_range: Range of k values to test (default: 2 to min(15, n_samples-1))
        output_file: HTML file to save the silhouette plot

    Returns:
        dict: Contains k_values, silhouette_scores, and suggested optimal k
    """

    if k_range is None:
        max_k = min(15, len(article_vectors) - 1)
        k_range = range(
            2, max_k + 1
        )  # Start from 2 since silhouette needs at least 2 clusters

    silhouette_scores = []

    print("Computing silhouette method...")
    for k in k_range:
        if k >= len(article_vectors):
            break

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(article_vectors)

        # Calculate silhouette score
        score = silhouette_score(article_vectors, cluster_labels)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette score={score:.3f}")

    # Find the optimal k (highest silhouette score)
    if silhouette_scores:
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[optimal_idx]
        optimal_score = silhouette_scores[optimal_idx]
    else:
        optimal_k = 2  # Default fallback
        optimal_score = 0.0

    # Create plotly visualization
    fig = go.Figure()

    k_values = list(k_range)[: len(silhouette_scores)]

    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=silhouette_scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
        )
    )

    # Highlight the suggested optimal k
    if optimal_k in k_values:
        fig.add_trace(
            go.Scatter(
                x=[optimal_k],
                y=[optimal_score],
                mode="markers",
                name=f"Optimal k={optimal_k} (score={optimal_score:.3f})",
                marker=dict(size=15, color="red", symbol="diamond"),
            )
        )

    fig.update_layout(
        title="Silhouette Method for Optimal Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        width=800,
        height=500,
        showlegend=True,
        yaxis=dict(
            range=[0, 1]
        ),  # Silhouette scores range from -1 to 1, but typically 0 to 1
    )

    # Save HTML plot
    fig.write_html(output_file)
    print(f"Silhouette plot saved to {output_file}")

    # Also save static PNG
    png_file = output_file.replace(".html", "_static.png")
    try:
        fig.write_image(png_file, width=800, height=500, scale=2)
        print(f"Static silhouette plot saved to {png_file}")
    except Exception as e:
        print(f"Could not save PNG (install kaleido with: pip install kaleido): {e}")

    print(
        f"Suggested optimal number of clusters: {optimal_k} "
        f"(silhouette score: {optimal_score:.3f})"
    )

    return {
        "k_values": k_values,
        "silhouette_scores": silhouette_scores,
        "optimal_k": optimal_k,
        "optimal_score": optimal_score,
    }


# ---------- CLUSTER ANALYSIS ----------
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
        top_sim_idx = np.argsort(sims)[-max_terms * 2 :]  # a bit more, then dedupe
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


# ---------- GEOMETRY UTILITIES ----------
def compute_cluster_ellipse(
    points_2d: np.ndarray, n_std: float = 2.0, n_points: int = 100
):
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
