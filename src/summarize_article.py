import os
import pandas as pd
from ollama import Client
from ollama import chat
from ollama import ChatResponse
from openai import OpenAI
import pathlib

import argparse

import ollama
import requests
import environ
env = environ.Env(
    DEBUG=(bool, False)
)

#TODO: Add Gemini, DeepSeek and Ollama
def summarize_article_ollama(prompt: str, model: str = "gemma3") -> str:
    OLLAMA_API_KEY = env('OLLAMA_API_KEY')
    url = 'http://llm.nina.no/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {OLLAMA_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": prompt
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    print(f"Response text: {response.text}")
    #return response.json()

    #response = ollama.generate(
    #    model=model,  # Replace with the actual name of your Ollama model
    #    prompt=prompt,
    #    options={
    #        'num_ctx': 200000  # Maximum number of tokens for the query
    #    }
    #)
    return response['response']

def summarize_article_gpt(prompt: str) -> str:
    OPENAI_API_KEY = env('OPENAI_API_KEY')
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_words(input_path: str, output_path: str, model: str = "gpt"):

    with open(input_path, "r", encoding="utf-8") as file:
        article = file.read()

    prompt = f"""Read the text. Identify the ten most essential conceptual terms 
    that capture the core ideas of the article. 
    
    - Use single words only (no phrases).
    - Pick concepts, not summaries or opinions.
    - Output a valid python list of exactly ten lowercase words.
    - Do not repeat near-synonyms:
    
    Here is the article: \n\n{article}"""

    if model == "gpt":
        summary = summarize_article_gpt(prompt)
    elif model == "gemma":
        summary = summarize_article_ollama(prompt, model="gemma3")
    elif model == "deepseek":
        summary = summarize_article_ollama(prompt, model="deepseek-r1")
    elif model == "mistral":
        summary = summarize_article_ollama(prompt, model="mistral")
    else:
        print("Select a valid option: gpt, gemma, deepseek, mistral")

    summary_pd = {
        'Article': os.path.basename(input_path),
        'Content': summary
         }

    df = pd.DataFrame([summary_pd])
    df.to_csv(output_path, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input markdown file")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the summary CSV")
    parser.add_argument("--model", type=str, default="gpt", help="Model to use for summarization")
    args = parser.parse_args()

    if args.output_path is None:
        filename = pathlib.Path(args.input_path).stem + "_summary.csv"
        output_dir = pathlib.Path(args.input_path).parent.parent / f"summaries_{args.model}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_path = str(output_dir / filename)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environ.Env.read_env(os.path.join(BASE_DIR, '.env'))
    
    get_words(input_path=args.input_path, output_path=args.output_path, model=args.model)
    # uv run python -m src.summarize_article --input_path "/home/benjamin.cretois/Code/social-science-AI/test/markdown_output/article.md" --model "gpt"