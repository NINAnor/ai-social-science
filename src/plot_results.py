"""
Main plotting script for analyzing and visualizing article concepts and clusters using UMAP.
"""
import argparse
import sys
import os
import json
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    merge_dfs,
    safe_parse_concepts,
    get_embeddings,
    compute_cluster_names,
)


def get_optimal_clusters(n_clusters=None, assets_dir="assets"):
    """Get optimal number of clusters from precomputed results or use provided value."""
    if n_clusters is not None:
        print(f"Using specified n_clusters={n_clusters}")
        return n_clusters
    
    # Try to load precomputed results
    results_file = os.path.join(assets_dir, "cluster_analysis_results.json")
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        optimal_k = results['optimal_k']
        optimal_score = results['optimal_score']
        print(f"Using precomputed optimal clusters: k={optimal_k} (silhouette score: {optimal_score:.3f})")
        return optimal_k
    except FileNotFoundError:
        print("No precomputed cluster analysis found. Using default n_clusters=5")
        print("(Run: uv run python src/find_optimal_clusters.py --summary_dir <dir> to find optimal clusters)")
        return 5


def create_umap_plot(article_ids, article_2d, cluster_labels, cluster_names, 
                     concept_2d=None, concept_labels=None, output_file="plot.html"):
    """Create interactive UMAP plot with clusters."""
    fig = go.Figure()
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Add articles
    fig.add_trace(
        go.Scatter(
            x=article_2d[:, 0], y=article_2d[:, 1], mode="markers",
            marker=dict(size=10, color=cluster_labels, colorscale="Viridis"),
            hovertext=[f"{aid}<br>Cluster {clust}: {cluster_names.get(clust, '')}"
                      for aid, clust in zip(article_ids, cluster_labels)],
            hoverinfo="text", name="Articles",
        )
    )

    # Add concepts if provided
    if concept_2d is not None and concept_labels is not None:
        fig.add_trace(
            go.Scatter(
                x=concept_2d[:, 0], y=concept_2d[:, 1], mode="markers",
                marker=dict(symbol="x", size=9), name="Key concepts",
                hovertext=concept_labels, hoverinfo="text",
            )
        )

    # Add cluster labels at centroids
    centroids = []
    centroid_texts = []
    for clust in unique_clusters:
        pts = article_2d[cluster_labels == clust]
        centroids.append(pts.mean(axis=0))
        centroid_texts.append(cluster_names.get(clust, f"Cluster {clust}"))

    if centroids:
        centroids = np.array(centroids)
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1], mode="text",
                text=centroid_texts, textposition="middle center",
                showlegend=False, hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Articles in UMAP Space",
        xaxis_title="UMAP1", yaxis_title="UMAP2",
        width=1000, height=700,
    )

    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

    # Save static PNG
    png_file = output_file.replace('.html', '_static.png')
    try:
        fig.write_image(png_file, width=1000, height=700, scale=2)
        print(f"Static plot saved to {png_file}")
    except Exception as e:
        print(f"Could not save PNG: {e}")


def create_concepts_plot(unique_concepts, concept_2d, output_file="concepts.html", concept_counts=None):
    """Create plot showing only concepts in UMAP space."""
    fig = go.Figure()
    
    if concept_counts is not None:
        # Show only the most frequent concepts to reduce clutter
        top_concepts = [word for word, _ in concept_counts.most_common(50)]
        mask = [c in top_concepts for c in unique_concepts]
        selected_concepts = [c for c, include in zip(unique_concepts, mask) if include]
        selected_coords = concept_2d[mask]
        
        # Size markers by frequency
        frequencies = [concept_counts[c] for c in selected_concepts]
        total_concepts = sum(concept_counts.values())
        max_freq = max(frequencies)
        sizes = [8 + (freq / max_freq) * 12 for freq in frequencies]  # Size 8-20
        
        fig.add_trace(
            go.Scatter(
                x=selected_coords[:, 0], y=selected_coords[:, 1],
                mode="markers+text",
                marker=dict(size=sizes, color="blue", opacity=0.6),
                text=selected_concepts, 
                textposition="top center",
                textfont=dict(size=12, color="darkblue"),
                hovertext=[f"{c} ({freq/total_concepts*100:.1f}%)" for c, freq in zip(selected_concepts, frequencies)],
                hoverinfo="text", 
                name="Top concepts",
            )
        )
        title_text = f"Top 50 Most Frequent Concepts in UMAP Space"
    else:
        # Fallback: show every nth concept
        step = max(1, len(unique_concepts) // 50)
        selected_indices = range(0, len(unique_concepts), step)
        selected_concepts = [unique_concepts[i] for i in selected_indices]
        selected_coords = concept_2d[selected_indices]
        
        fig.add_trace(
            go.Scatter(
                x=selected_coords[:, 0], y=selected_coords[:, 1],
                mode="markers+text",
                marker=dict(size=10, color="blue", opacity=0.6),
                text=selected_concepts, 
                textposition="top center",
                textfont=dict(size=12, color="darkblue"),
                hovertext=selected_concepts,
                hoverinfo="text", 
                name="Concepts",
            )
        )
        title_text = f"Concepts in UMAP Space (showing {len(selected_concepts)} of {len(unique_concepts)})"
    
    # Add all concepts as small dots for context
    fig.add_trace(
        go.Scatter(
            x=concept_2d[:, 0], y=concept_2d[:, 1],
            mode="markers",
            marker=dict(size=3, color="lightgray", opacity=0.2),
            hovertext=unique_concepts,
            hoverinfo="text",
            name="All concepts",
            showlegend=False
        )
    )
    
    fig.update_layout(
        title=title_text,
        xaxis_title="UMAP1", yaxis_title="UMAP2",
        width=1200, height=800, showlegend=False
    )
    
    fig.write_html(output_file)
    print(f"Concepts plot saved to {output_file}")

    # Save static PNG
    png_file = output_file.replace('.html', '_static.png')
    try:
        fig.write_image(png_file, width=1200, height=800, scale=2)
        print(f"Static concepts plot saved to {png_file}")
    except Exception as e:
        print(f"Could not save PNG: {e}")


def main(summary_dir: str, output_html: str, n_clusters=None):
    """Main entry point for the script."""
    # Create assets directory if it doesn't exist
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # Update output paths to use assets directory
    if not output_html.startswith(assets_dir):
        output_html = os.path.join(assets_dir, os.path.basename(output_html))
    
    # Load data
    df = merge_dfs(summary_dir)
    articles_concepts = {}
    for _, row in df.iterrows():
        concepts = safe_parse_concepts(row["Content"])
        if (isinstance(concepts, list) and len(concepts) > 0 and len(concepts) <= 10 
            and all(isinstance(w, str) and ' ' not in w for w in concepts)):
            articles_concepts[row["Article"]] = concepts

    print(f"Processed {len(articles_concepts)} articles with valid concepts")

    # Get embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    article_ids, article_vectors = get_embeddings(model, articles_concepts)
    
    # Get all unique concepts and their embeddings
    unique_concepts = sorted({c for lst in articles_concepts.values() for c in lst})
    concept_embs = model.encode(unique_concepts)

    # UMAP transformation
    reducer = umap.UMAP(n_components=2, random_state=42)
    article_2d = reducer.fit_transform(article_vectors)
    concept_2d = reducer.transform(concept_embs)

    # Get top frequent concepts for display
    concept_counts = Counter()
    for concepts in articles_concepts.values():
        concept_counts.update(concepts)
    
        top_concepts = [word for word, _ in concept_counts.most_common(50)]
    mask = [c in top_concepts for c in unique_concepts]
    concept_2d_filtered = concept_2d[mask]
    concept_labels_filtered = [c for c in unique_concepts if c in top_concepts]

    # Clustering
    n_clusters = get_optimal_clusters(n_clusters, assets_dir)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(article_vectors)

    print(f"\nArticles per cluster:")
    for cl in np.unique(cluster_labels):
        print(f"  Cluster {cl}: {(cluster_labels == cl).sum()} articles")

    # Get cluster names
    cluster_names = compute_cluster_names(
        article_ids, article_vectors, articles_concepts, cluster_labels,
        unique_concepts, concept_embs, max_terms=5,
    )

    print(f"\nCluster themes:")
    for cl in sorted(cluster_names.keys()):
        print(f"  Cluster {cl}: {cluster_names[cl]}")

    # Create plots
    create_umap_plot(article_ids, article_2d, cluster_labels, cluster_names, 
                     concept_2d_filtered, concept_labels_filtered, output_html)
    
    concepts_only_html = output_html.replace('.html', '_concepts_only.html')
    create_concepts_plot(top_concepts, concept_2d[mask], concepts_only_html, concept_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UMAP visualization of article clusters")
    parser.add_argument("--summary_dir", required=True, help="Directory containing article summaries")
    parser.add_argument("--output_html", default="articles_umap_plot.html", help="Output HTML file")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters (auto-detected if not specified)")
    
    args = parser.parse_args()
    main(args.summary_dir, args.output_html, args.n_clusters)
