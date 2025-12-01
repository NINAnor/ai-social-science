"""
Main plotting script for analyzing and visualizing article concepts and clusters.
"""
import argparse
import sys
import os

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    merge_dfs,
    safe_parse_concepts,
    get_embeddings,
    interpret_axes,
    build_axis_labels,
    silhouette_method,
    compute_cluster_names,
    compute_cluster_ellipse,
)
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
    """Create an interactive Plotly visualization of the PCA results with clusters."""
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
    
    # Also save a static PNG for README
    png_file = output_file.replace('.html', '_static.png')
    try:
        fig.write_image(png_file, width=1000, height=700, scale=2)
        print(f"Static plot saved to {png_file}")
    except Exception as e:
        print(f"Could not save PNG (install kaleido with: pip install kaleido): {e}")


def plot_articles_and_concepts(
    article_ids,
    article_vectors,
    articles_concepts,
    model,
    n_clusters=None,
    use_silhouette_method=True,
    top_axis_terms=10,
    output_html="articles_pca_plot.html",
):
    """Main function to perform PCA, clustering, and visualization."""
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

    # Determine optimal number of clusters
    if use_silhouette_method and n_clusters is None:
        print("\n" + "="*50)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*50)
        silhouette_result = silhouette_method(article_vectors, output_file="silhouette_plot.html")
        n_clusters = silhouette_result['optimal_k']
        print(f"Using optimal k={n_clusters} from silhouette method")
    elif n_clusters is None:
        n_clusters = 5  # Default fallback
        print(f"Using default n_clusters={n_clusters}")

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


def main(summary_dir: str, output_html: str, n_clusters=None, use_silhouette_method=True):
    """Main entry point for the script."""
    df = merge_dfs(summary_dir)

    articles_concepts = {}
    for _, row in df.iterrows():
        try:
            concepts = safe_parse_concepts(row["Content"])
            if isinstance(concepts, list) and len(concepts) == 10:
                articles_concepts[row["Article"]] = concepts
                print(concepts)
        except Exception as e:
            print(f"Warning: Could not parse concepts for {row['Article']}: {e}")
            continue

    model = SentenceTransformer("all-MiniLM-L6-v2")
    article_ids, article_vectors = get_embeddings(model, articles_concepts)

    plot_articles_and_concepts(
        article_ids,
        article_vectors,
        articles_concepts,
        model,
        n_clusters=n_clusters,
        use_silhouette_method=use_silhouette_method,
        output_html=output_html,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument(
        "--output_html", default="articles_pca_plot.html"
    )
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=None, 
        help="Number of clusters (if not specified, silhouette method will be used)"
    )
    parser.add_argument(
        "--no_silhouette_method", 
        action="store_true", 
        help="Disable silhouette method and use default n_clusters=5 if not specified"
    )
    args = parser.parse_args()

    main(
        args.summary_dir, 
        args.output_html, 
        n_clusters=args.n_clusters,
        use_silhouette_method=not args.no_silhouette_method
    )
