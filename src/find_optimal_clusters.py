"""
Script for finding optimal number of clusters using various methods.
This script is focused solely on cluster optimization and can be run independently.
"""
import argparse
import sys
import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    merge_dfs,
    safe_parse_concepts,
    get_embeddings,
    silhouette_method,
)


def analyze_optimal_clusters(
    article_vectors, 
    k_range=None, 
    output_dir="assets",
    save_results=True
):
    """
    Analyze optimal number of clusters using silhouette method.
    
    Args:
        article_vectors: The embedding vectors for articles
        k_range: Range of k values to test (default: 2 to min(15, n_samples-1))
        output_dir: Directory to save plots and results
        save_results: Whether to save results to JSON file
    
    Returns:
        dict: Contains analysis results including optimal k and scores
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("CLUSTER OPTIMIZATION ANALYSIS")
    print("="*50)
    print(f"Number of articles: {len(article_vectors)}")
    print(f"Vector dimensions: {article_vectors.shape[1]}")
    print()
    
    # Run silhouette analysis
    silhouette_plot_path = os.path.join(output_dir, "silhouette_plot.html")
    silhouette_results = silhouette_method(
        article_vectors, 
        k_range=k_range, 
        output_file=silhouette_plot_path
    )
    
    # Save results to JSON for later use
    if save_results:
        results_file = os.path.join(output_dir, "cluster_analysis_results.json")
        with open(results_file, 'w') as f:
            json.dump(silhouette_results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    optimal_k = silhouette_results['optimal_k']
    optimal_score = silhouette_results['optimal_score']
    print(f"Recommended number of clusters: {optimal_k}")
    print(f"Silhouette score: {optimal_score:.3f}")
    print(f"Silhouette plot: {silhouette_plot_path}")
    if save_results:
        print(f"Detailed results: {results_file}")
    
    return silhouette_results

def main(summary_dir: str, output_dir: str = "assets", k_range: str = None):
    
    if k_range:
        try:
            if '-' in k_range:
                start, end = map(int, k_range.split('-'))
                k_range = range(start, end + 1)
            else:
                k_values = [int(x.strip()) for x in k_range.split(',')]
                k_range = k_values
        except ValueError:
            print(f"Invalid k_range format: {k_range}")
            print("Use format like '2-15' or '2,3,4,5,6'")
            sys.exit(1)
    
    # Load and process data
    print("Loading article data...")
    df = merge_dfs(summary_dir)

    articles_concepts = {}
    processed_count = 0
    for _, row in df.iterrows():
        concepts = safe_parse_concepts(row["Content"])
        if (
            isinstance(concepts, list)
            and len(concepts) > 0
            and len(concepts) <= 10
            and all(isinstance(w, str) and ' ' not in w for w in concepts)
        ):
            articles_concepts[row["Article"]] = concepts
            processed_count += 1

    model = SentenceTransformer("all-MiniLM-L6-v2")
    article_ids, article_vectors = get_embeddings(model, articles_concepts)
    
    # Analyze optimal clusters
    results = analyze_optimal_clusters(
        article_vectors,
        k_range=k_range,
        output_dir=output_dir,
        save_results=True
    )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find optimal number of clusters for article analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - analyze clusters 2-15
  python src/find_optimal_clusters.py --summary_dir summaries_deepseek-r1_trial1
    """
    )
    parser.add_argument(
        "--summary_dir", 
        required=True, 
        help="Directory containing article summaries"
    )
    parser.add_argument(
        "--output_dir", 
        default="assets", 
        help="Directory to save plots and results (default: assets)"
    )
    parser.add_argument(
        "--k_range", 
        default=None, 
        help="Range of k values to test. Format: '2-15' or '2,3,4,5,6' (default: 2 to min(15, n_articles-1))"
    )
    
    args = parser.parse_args()
    
    main(args.summary_dir, args.output_dir, args.k_range)