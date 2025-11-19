import pathlib

import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

import argparse
import pandas as pd


def merge_dfs(summary_dir: str) -> pd.DataFrame:
    summary_files = pathlib.Path(summary_dir).glob("*.csv")
    df_list = [pd.read_csv(f) for f in summary_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

def get_embeddings(model, df: pd.DataFrame) -> np.ndarray:

    article_ids = []
    article_vectors = []

    for art_id, concepts in df.items():
        concept_embs = model.encode(concepts)
        # Average to get one vector per article
        article_vec = np.mean(concept_embs, axis=0)
        article_ids.append(art_id)
        article_vectors.append(article_vec)

    article_vectors = np.vstack(article_vectors)  
    return article_ids, article_vectors

def plotly_plot(article_ids, article_2d, concept_2d, unique_concepts, output_file="plot.html"):
    fig = go.Figure()

    # Articles
    fig.add_trace(go.Scatter(
        x=article_2d[:, 0],
        y=article_2d[:, 1],
        mode="markers",
        marker=dict(size=12),
        name="Articles",
        hovertext=article_ids,
        hoverinfo="text"
    ))

    # Concepts
    fig.add_trace(go.Scatter(
        x=concept_2d[:, 0],
        y=concept_2d[:, 1],
        mode="markers",
        marker=dict(symbol="x", size=10),
        name="Concepts",
        hovertext=unique_concepts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title="Articles and Concepts in PCA Space",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=800,
        height=600
    )

    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")
    
def matplot_plot(article_ids, article_2d, concept_2d, unique_concepts, model, pca):

    plt.figure(figsize=(6, 5))
    plt.scatter(article_2d[:, 0], article_2d[:, 1])

    for i, art_id in enumerate(article_ids):
        plt.text(article_2d[i, 0], article_2d[i, 1], art_id)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Articles in concept space (PCA of concept embeddings)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.scatter(article_2d[:, 0], article_2d[:, 1], marker="o", label="Articles")
    plt.scatter(concept_2d[:, 0], concept_2d[:, 1], marker="x", alpha=0.6, label="Concepts")

    # Labels
    for i, art_id in enumerate(article_ids):
        plt.text(article_2d[i, 0], article_2d[i, 1], art_id)

    for i, word in enumerate(unique_concepts):
        plt.text(concept_2d[i, 0], concept_2d[i, 1], word, fontsize=8, alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Articles and concepts in shared PCA space")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("articles_concepts_pca.png", dpi=300)

def main(summary_dir: str, output_image: str = "articles_concepts_pca.png"):
    articles_concepts = {
        "A1": ["biodiversity", "ecosystem", "conservation", "habitat", "species",
            "restoration", "fragmentation", "landscape", "connectivity", "resilience"],
        "A2": ["machine learning", "deep learning", "neural networks", "classification",
            "prediction", "regression", "optimization", "training", "testing", "validation"],
        "A3": ["climate change", "global warming", "carbon emissions", "mitigation",
            "adaptation", "temperature", "precipitation", "extreme events", "IPCC", "scenarios"],
        "A4": ["forest", "logging", "deforestation", "land use", "carbon storage",
            "biodiversity", "ecosystem services", "policy", "management", "governance"],
        # ... add more articles
    }

    unique_concepts = sorted({c for concepts in articles_concepts.values() for c in concepts})

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast; good enough for this

    # Compute embeddings
    article_ids, article_vectors = get_embeddings(model, articles_concepts)
    concept_embs = model.encode(unique_concepts)

    # Do the PCA
    pca = PCA(n_components=2)
    article_2d = pca.fit_transform(article_vectors)  # shape: (N_articles, 2)
    concept_2d = pca.transform(concept_embs)

    plotly_plot(article_ids, article_2d, concept_2d, unique_concepts, "articles_pca_plot.html")
    matplot_plot(article_ids, article_2d, concept_2d, unique_concepts, model, pca)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", type=str, required=True, help="Path to the csv summaries") 
    parser.add_argument("--output_image", type=str, required=False, default="articles_concepts_pca.png", help="Name of the output image file")
    args = parser.parse_args()

    main(args.summary_dir, args.output_image)