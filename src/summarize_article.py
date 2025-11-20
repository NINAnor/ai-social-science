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
        options={"num_ctx": 30000},
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

    prompt = f"""Read the text. Identify the ten most essential conceptual terms
    that capture the core ideas of the article.

    - Use single words only (no phrases).
    - Pick concepts, not summaries or opinions.
    - Output a valid python list of exactly ten lowercase words.
    - Do not repeat near-synonyms
    - Output the python list only, no explanation, reasoning or additional text.

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
