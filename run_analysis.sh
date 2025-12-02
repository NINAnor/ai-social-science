#!/bin/bash

# Article Analysis Workflow
# This script demonstrates the recommended workflow for analyzing articles
# using the decoupled cluster optimization and PCA plotting scripts.

set -e  # Exit on any error

# Configuration
SUMMARY_DIR=${1:-"summaries_deepseek-r1_trial1"}
ASSETS_DIR="assets"
K_RANGE=${2:-"2-15"}

echo "==============================================="
echo "Article Analysis Workflow"
echo "==============================================="
echo "Summary directory: $SUMMARY_DIR"
echo "Assets directory: $ASSETS_DIR"
echo "K-range for cluster analysis: $K_RANGE"
echo ""

# Check if summary directory exists
if [ ! -d "$SUMMARY_DIR" ]; then
    echo "Error: Summary directory '$SUMMARY_DIR' not found"
    echo "Usage: $0 <summary_dir> [k_range]"
    echo "Example: $0 summaries_deepseek-r1_trial1 2-10"
    exit 1
fi

echo "Step 1: Running cluster optimization analysis..."
echo "==============================================="
uv run python src/find_optimal_clusters.py \
    --summary_dir "$SUMMARY_DIR" \
    --k_range "$K_RANGE" \
    --output_dir "$ASSETS_DIR"

echo ""
echo "Step 2: Generating PCA visualization..."
echo "==============================================="
uv run python src/plot_results.py \
    --summary_dir "$SUMMARY_DIR"

echo ""
echo "==============================================="
echo "Analysis complete! Check the following files:"
echo "==============================================="
echo "📊 Interactive PCA plot: $ASSETS_DIR/articles_pca_plot.html"
echo "📈 Static PCA plot: $ASSETS_DIR/articles_pca_plot_static.png"
echo "📊 Interactive silhouette plot: $ASSETS_DIR/silhouette_plot.html"
echo "📈 Static silhouette plot: $ASSETS_DIR/silhouette_plot_static.png"
echo "📋 Cluster analysis results: $ASSETS_DIR/cluster_analysis_results.json"
echo ""
echo "💡 To rerun just the PCA plot with different parameters:"
echo "   uv run python src/plot_results.py --summary_dir $SUMMARY_DIR --n_clusters <N>"
echo ""
echo "💡 To recompute cluster analysis with different k-range:"
echo "   uv run python src/find_optimal_clusters.py --summary_dir $SUMMARY_DIR --k_range <range>"