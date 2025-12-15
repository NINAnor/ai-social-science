import argparse
import pathlib

import environ
import ollama
import pandas as pd
from ollama import Client
from openai import OpenAI

env = environ.Env(DEBUG=(bool, False))


def summarize_article_ollama(prompt: str, model: str) -> str:
    ollama.pull(model)

    client = Client(
        host="http://localhost:11436"
        # headers={'Authorization': 'Bearer ' + env('OLLAMA_API_KEY')}
    )
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 35000},
    )

    print()
    return response.message.content


def summarize_article_gpt(prompt: str) -> str:
    OPENAI_API_KEY = env("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_words(input_path: str, output_path: str, model: str = "gpt"):
    input_file = pathlib.Path(input_path)
    with input_file.open(encoding="utf-8") as file:
        article = file.read()

    prompt = f"""Read the text. Identify the concepts that appear in the semantic \
neighbourhood of the idea of "social acceptance" and
    its close synonyms (public acceptance, community acceptance, societal \
acceptance, stakeholder acceptance, licence to operate, social licence).
        - Use only single words.
        - Do not use "acceptance" and its morphological variants \
(acceptability, accepted, etc.)
        - Output a Python list of 0–10 lowercase words.
        - Include only concepts that relate to how the article uses, defines, \
or implies social acceptance.
        - Ignore concepts unrelated to social acceptance.
        - If the text contains no content related to social acceptance or its \
close synonyms, output an empty list [] and do not infer or guess.
        - Only output words that occur explicitly in the text or are directly \
implied by its discussion of acceptance.
        - Do not add speculative concepts.

    Here is the article: \n\n{article}"""

    if model == "gpt":
        summary = summarize_article_gpt(prompt)
    else:
        summary = summarize_article_ollama(prompt, model=model)

    summary_pd = {"Article": pathlib.Path(input_path).name, "Content": summary}

    df = pd.DataFrame([summary_pd])
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input markdown file"
    )
    parser.add_argument(
        "--output_path", type=str, required=False, help="Path to save the summary CSV"
    )
    parser.add_argument(
        "--model", type=str, default="gpt", help="Model to use for summarization"
    )
    args = parser.parse_args()

    if args.output_path is None:
        filename = pathlib.Path(args.input_path).stem + "_summary.csv"
        output_dir = (
            pathlib.Path(args.input_path).parent.parent / f"summaries_{args.model}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_path = str(output_dir / filename)

    BASE_DIR = pathlib.Path(__file__).parent.parent
    environ.Env.read_env(BASE_DIR / ".env")

    get_words(
        input_path=args.input_path, output_path=args.output_path, model=args.model
    )
    # uv run python -m src.summarize_article --input_path \
    # "/home/benjamin.cretois/Code/social-science-AI/test/markdown_output/article.md" \
    # --model "gpt"
