"""
Main plotting script for analyzing and visualizing article concepts and clusters.
This script focuses on PCA visualization and loads pre-computed cluster analysis results.
For cluster optimization, use the separate find_optimal_clusters.py script.
"""
import argparse
import sys
import os
import json

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
    compute_cluster_names,
    compute_cluster_ellipse,
)


def load_cluster_analysis_results(results_file="assets/cluster_analysis_results.json"):
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded cluster analysis results from {results_file}")
        return results
    except FileNotFoundError:
        print(f"No pre-computed cluster analysis found at {results_file}")
        return None


def determine_optimal_clusters(
    n_clusters=None, 
    assets_dir="assets",
    load_precomputed=True
):

    if n_clusters is not None:
        print(f"Using explicitly specified n_clusters={n_clusters}")
        return n_clusters
    
    if load_precomputed:
        results_file = os.path.join(assets_dir, "cluster_analysis_results.json")
        precomputed_results = load_cluster_analysis_results(results_file)
        if precomputed_results:
            optimal_k = precomputed_results['optimal_k']
            optimal_score = precomputed_results['optimal_score']
            print(f"Using precomputed optimal clusters: k={optimal_k} (silhouette score: {optimal_score:.3f})")
            print(f"(If you want to recompute, run: python src/find_optimal_clusters.py --summary_dir <dir>)")
            return optimal_k

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
    top_axis_terms=10,
    output_html="articles_pca_plot.html",
    assets_dir="assets",
    load_precomputed_clusters=True,
):

    unique_concepts = sorted({c for lst in articles_concepts.values() for c in lst})
    concept_embs = model.encode(unique_concepts)

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

    # Determine optimal number of clusters using the new centralized logic
    n_clusters = determine_optimal_clusters(
        n_clusters=n_clusters,
        assets_dir=assets_dir,
        load_precomputed=load_precomputed_clusters
    )

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


def main(summary_dir: str, output_html: str, n_clusters=None, load_precomputed_clusters=True):
    """Main entry point for the script."""
    import os
    
    # Create assets directory if it doesn't exist
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # Update output paths to use assets directory
    if not output_html.startswith(assets_dir):
        output_html = os.path.join(assets_dir, os.path.basename(output_html))
    
    df = merge_dfs(summary_dir)

    articles_concepts = {}
    for _, row in df.iterrows():
        concepts = safe_parse_concepts(row["Content"])

        if (
            isinstance(concepts, list)
            and len(concepts) > 0
            and len(concepts) <= 10
            and all(isinstance(w, str) and ' ' not in w for w in concepts)
        ):
            articles_concepts[row["Article"]] = concepts
            print(concepts)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    article_ids, article_vectors = get_embeddings(model, articles_concepts)

    plot_articles_and_concepts(
        article_ids,
        article_vectors,
        articles_concepts,
        model,
        n_clusters=n_clusters,
        output_html=output_html,
        assets_dir=assets_dir,
        load_precomputed_clusters=load_precomputed_clusters,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PCA visualization of article clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (tries to load precomputed cluster analysis)
  python src/plot_results.py --summary_dir summaries_deepseek-r1_trial1
  
  # Use specific number of clusters
  python src/plot_results.py --summary_dir summaries_deepseek-r1_trial1 --n_clusters 5
  
  # Skip loading precomputed results (use default n_clusters=5)
  python src/plot_results.py --summary_dir summaries_deepseek-r1_trial1 --no_load_precomputed

Recommended workflow:
  1. First run: python src/find_optimal_clusters.py --summary_dir <dir>
  2. Then run:  python src/plot_results.py --summary_dir <dir>
        """
    )
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument(
        "--output_html", default="articles_pca_plot.html"
    )
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=None, 
        help="Number of clusters (if not specified, will try to load precomputed results)"
    )
    parser.add_argument(
        "--no_load_precomputed",
        action="store_true",
        help="Don't try to load precomputed cluster analysis results"
    )
    args = parser.parse_args()

    main(
        args.summary_dir, 
        args.output_html, 
        n_clusters=args.n_clusters,
        load_precomputed_clusters=not args.no_load_precomputed
    )
