# AI-social-science

This is the repository for AI and social science: how can AI help social scientists in the litterature review process.

The steps are described below:

```mermaid

flowchart LR
    A[1. PDF article from Zotero] -->|article_to_markdown.py| B(2. Article in markdown)
    B --> |summarize_article.py| C(3. 10 words summary)
    C --> |plot_results.py| D(4. Plot 2D PCA of the 10 words)
```

## 1. PDF article from Zotero

After suggesting a search string, we start the search in Scopus and Google Scholar and extract the most relevant articles in Zotero. Using Zotero extract function, we pull the PDF of the articles. 

**_NOTE_**: Not all PDFs are found and there may be a need for manually extracting the articles.

## 2. Article in markdown

PDF is made for humans, but not so much for machines. Before inputting the text into a LLM, we need to turn the PDF into a markdown file. For this we use the script `articles_to_markdown.py` which queries a deep learning vision model that takes the pages of the PDF as input and output its text as a markdown document.

**_NOTE_**: Here we are using gpt-4o-mini 

## 3. 10 words summary

Now that the text is machine readable, we input the markdown article to an LLM and ask it to summarize the article in 10 conceptual words. The summary is a python list that is saved as a .csv for the next step.

The prompt is as follow:

```
    prompt = f"""Read the text. Identify the ten most essential conceptual terms
    that capture the core ideas of the article.

    - Use single words only (no phrases).
    - Pick concepts, not summaries or opinions.
    - Output a valid python list of exactly ten lowercase words.
    - Do not repeat near-synonyms
    - Output the python list only, no explanation, reasoning or additional text.

    Here is the article: \n\n{article}"""
```

## 4. Plot 2D PCA of the 10 words

Finally, we use an embedding model to turn the list of 10 words into a single vector. This vector is a numerical representation of the concepts found in the article. We plot the vectors of each article in a 2D plot using a PCA.