"""
Main plotting script for analyzing and visualizing article concepts and
clusters using UMAP.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Add the parent directory to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    compute_cluster_names,
    get_embeddings,
    merge_dfs,
    safe_parse_concepts,
)


def get_optimal_clusters(n_clusters=None, assets_dir="assets"):
    """Get optimal number of clusters from precomputed results or use provided value."""
    if n_clusters is not None:
        print(f"Using specified n_clusters={n_clusters}")
        return n_clusters

    # Try to load precomputed results
    results_file = Path(assets_dir) / "cluster_analysis_results.json"
    try:
        results = json.loads(results_file.read_text())
        optimal_k = results["optimal_k"]
        optimal_score = results["optimal_score"]
        print(
            f"Using precomputed optimal clusters: k={optimal_k} "
            f"(silhouette score: {optimal_score:.3f})"
        )
        return optimal_k
    except FileNotFoundError:
        print("No precomputed cluster analysis found. Using default n_clusters=5")
        print(
            "(Run: uv run python src/find_optimal_clusters.py "
            "--summary_dir <dir> to find optimal clusters)"
        )
        return 5


def create_umap_plot(
    article_ids,
    article_2d,
    cluster_labels,
    cluster_names,
    concept_2d=None,
    concept_labels=None,
    output_file="plot.html",
):
    """Create interactive UMAP plot with clusters."""
    fig = go.Figure()

    unique_clusters = np.unique(cluster_labels)

    # Add articles
    fig.add_trace(
        go.Scatter(
            x=article_2d[:, 0],
            y=article_2d[:, 1],
            mode="markers",
            marker=dict(size=10, color=cluster_labels, colorscale="Viridis"),
            hovertext=[
                f"{aid}<br>Cluster {clust}: {cluster_names.get(clust, '')}"
                for aid, clust in zip(article_ids, cluster_labels, strict=False)
            ],
            hoverinfo="text",
            name="Articles",
        )
    )

    # Add concepts if provided
    if concept_2d is not None and concept_labels is not None:
        fig.add_trace(
            go.Scatter(
                x=concept_2d[:, 0],
                y=concept_2d[:, 1],
                mode="markers",
                marker=dict(symbol="x", size=9),
                name="Key concepts",
                hovertext=concept_labels,
                hoverinfo="text",
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
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="text",
                text=centroid_texts,
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Articles in UMAP Space",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        width=1000,
        height=700,
    )

    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

    # Save static PNG
    png_file = output_file.replace(".html", "_static.png")
    try:
        fig.write_image(png_file, width=1000, height=700, scale=2)
        print(f"Static plot saved to {png_file}")
    except Exception as e:
        print(f"Could not save PNG: {e}")


def create_concepts_plot(
    unique_concepts, concept_2d, output_file="concepts.html", concept_counts=None
):
    """Create plot showing only concepts in UMAP space."""
    fig = go.Figure()

    if concept_counts is not None:
        # Show only the most frequent concepts to reduce clutter
        top_concepts = [word for word, _ in concept_counts.most_common(50)]
        mask = [c in top_concepts for c in unique_concepts]
        selected_concepts = [
            c for c, include in zip(unique_concepts, mask, strict=False) if include
        ]
        selected_coords = concept_2d[mask]

        # Size markers by frequency
        frequencies = [concept_counts[c] for c in selected_concepts]
        total_concepts = sum(concept_counts.values())
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        sizes = [8 + (freq / max_freq) * 12 for freq in frequencies]  # Size 8-20

        # Implement anti-overlap strategy: use alternating text positions
        text_positions = ["top center", "bottom center", "middle right", "middle left"] * (len(selected_concepts) // 4 + 1)

        # Add main scatter plot with markers and text labels
        fig.add_trace(
            go.Scatter(
                x=selected_coords[:, 0],
                y=selected_coords[:, 1],
                mode="markers+text",
                marker=dict(
                    size=sizes, 
                    color=frequencies,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(
                        title="Frequency of occurrence (%)",
                        tickvals=[min_freq, (min_freq + max_freq) / 2, max_freq],
                        ticktext=[f"{min_freq/total_concepts*100:.1f}%", 
                                f"{((min_freq + max_freq) / 2)/total_concepts*100:.1f}%",
                                f"{max_freq/total_concepts*100:.1f}%"],
                        x=1.05,  # Move colorbar closer to main figure
                        len=0.7   # Make colorbar shorter
                    )
                ),
                text=selected_concepts,
                textposition=text_positions[:len(selected_concepts)],
                textfont=dict(
                    size=18, 
                    color="black", 
                    family="Arial Black",
                    # Add text outline for better visibility
                ),
                hovertext=[
                    f"{c} ({freq / total_concepts * 100:.1f}%)"
                    for c, freq in zip(selected_concepts, frequencies, strict=False)
                ],
                hoverinfo="text",
                name="Concepts",
                showlegend=False
            )
        )
        
        # Text is now shown directly on markers, no need for separate annotations
        title_text = "Top 50 Most Frequent Concepts in UMAP Space"
    else:
        # Fallback: show every nth concept
        step = max(1, len(unique_concepts) // 50)
        selected_indices = range(0, len(unique_concepts), step)
        selected_concepts = [unique_concepts[i] for i in selected_indices]
        selected_coords = concept_2d[selected_indices]

        fig.add_trace(
            go.Scatter(
                x=selected_coords[:, 0],
                y=selected_coords[:, 1],
                mode="markers+text",
                marker=dict(size=10, color="blue", opacity=0.6),
                text=selected_concepts,
                textposition="top center",
                textfont=dict(size=18, color="black", family="Arial Black"),
                hovertext=selected_concepts,
                hoverinfo="text",
                name="Concepts",
            )
        )
        
        # Text is now shown directly on markers
        title_text = (
            f"Concepts in UMAP Space "
            f"(showing {len(selected_concepts)} of {len(unique_concepts)})"
        )

    # Size legend removed as requested

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20, family="Arial"),  # Increased from 16 to 20
            x=0.5
        ),
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        font=dict(size=20),  # Set global font size for axis ticks
        xaxis=dict(
            title=dict(font=dict(size=20)),  # Axis title font size
            tickfont=dict(size=20)           # Axis tick font size
        ),
        yaxis=dict(
            title=dict(font=dict(size=20)),  # Axis title font size
            tickfont=dict(size=20)           # Axis tick font size
        ),
        width=1400,  # Reduced since no legend needed
        height=900,
        showlegend=False,

        annotations=[
            dict(
                text="Dot size and color represent frequency of occurrence in the corpus.",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="darkblue",
                borderwidth=1
            )
        ] if concept_counts is not None else []
    )

    fig.write_html(output_file)
    print(f"Concepts plot saved to {output_file}")

    # Save static PNG with all concepts labeled and zoomed view
    png_file = output_file.replace(".html", "_static.png")
    try:
        # Create a copy of the figure for static version with all labels
        static_fig = go.Figure(fig)
        
        # Clear existing annotations and add all concepts for static version
        static_fig.layout.annotations = []
        
        if concept_counts is not None:
            # Add all concepts for static version with better positioning
            import random
            import math
            random.seed(42)
            
            for i, concept in enumerate(selected_concepts):
                coord = selected_coords[i]
                freq_pct = frequencies[i] / total_concepts * 100
                
                # Use more spread out positioning for static version
                angle = (i * (360 / len(selected_concepts))) * (math.pi / 180)
                base_distance = 20  # Smaller distance for static to fit more
                
                x_offset = base_distance * math.cos(angle)
                y_offset = base_distance * math.sin(angle)
                
                # Add randomization but smaller for static
                x_offset += random.uniform(-5, 5)
                y_offset += random.uniform(-5, 5)
                
                static_fig.add_annotation(
                    x=coord[0],
                    y=coord[1],
                    text=f"{concept}<br>({freq_pct:.1f}%)",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=0.8,
                    arrowwidth=1,
                    arrowcolor="rgba(100,100,100,0.5)",
                    ax=x_offset,
                    ay=y_offset,
                    font=dict(size=7, color="darkblue", family="Arial"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="darkblue",
                    borderwidth=0.5,
                    align="center"
                )
        
        # Set zoom to x=2 to x=12 for better visualization
        static_fig.update_xaxes(range=[2, 12])
        # Auto-adjust y-axis to maintain aspect ratio in the zoomed region
        static_fig.update_yaxes(range=[-4, 4])  # Adjust as needed
        
        # Add title note about zoom
        current_title = static_fig.layout.title.text if static_fig.layout.title else title_text
        static_fig.update_layout(
            title=f"{current_title}<br><sub>Static version: All concepts shown, zoomed view (x=2 to x=12)</sub>",
            width=1400,
            height=1000  # Increased height for better readability
        )
        
        static_fig.write_image(png_file, width=1400, height=1000, scale=2)
        print(f"Static concepts plot saved to {png_file}")
    except Exception as e:
        print(f"Could not save PNG: {e}")


def main(summary_dir: str, output_html: str, n_clusters=None):
    """Main entry point for the script."""
    # Create assets directory if it doesn't exist
    assets_dir = "assets"
    Path(assets_dir).mkdir(parents=True, exist_ok=True)

    # Update output paths to use assets directory
    if not output_html.startswith(assets_dir):
        output_html = str(Path(assets_dir) / Path(output_html).name)

    # Load data
    df = merge_dfs(summary_dir)
    articles_concepts = {}
    for _, row in df.iterrows():
        concepts = safe_parse_concepts(row["Content"])
        if (
            isinstance(concepts, list)
            and len(concepts) > 0
            and len(concepts) <= 10
            and all(isinstance(w, str) and " " not in w for w in concepts)
        ):
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

    print("\nArticles per cluster:")
    for cl in np.unique(cluster_labels):
        print(f"  Cluster {cl}: {(cluster_labels == cl).sum()} articles")

    # Get cluster names
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

    # Create plots
    create_umap_plot(
        article_ids,
        article_2d,
        cluster_labels,
        cluster_names,
        concept_2d_filtered,
        concept_labels_filtered,
        output_html,
    )

    concepts_only_html = output_html.replace(".html", "_concepts_only.html")
    create_concepts_plot(
        top_concepts, concept_2d[mask], concepts_only_html, concept_counts
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate UMAP visualization of article clusters"
    )
    parser.add_argument(
        "--summary_dir", required=True, help="Directory containing article summaries"
    )
    parser.add_argument(
        "--output_html", default="articles_umap_plot.html", help="Output HTML file"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        help="Number of clusters (auto-detected if not specified)",
    )

    args = parser.parse_args()
    main(args.summary_dir, args.output_html, args.n_clusters)
