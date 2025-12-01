import argparse
import asyncio
import json
import pathlib

import environ
from pyzerox import zerox

env = environ.Env(DEBUG=(bool, False))


async def to_markdown(
    file_path: str = None,
    output_dir: str = None,
    model: str = None,
    custom_system_prompt: str | None = None,
    select_pages: int | list[int] | None = None,
    kwargs=None,
):
    if kwargs is None:
        kwargs = {}
    result = await zerox(
        file_path=file_path,
        model=model,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
        select_pages=select_pages,
        **kwargs,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the input file"
    ) 
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to save the markdown output",
    ) 
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Model to use for conversion"
    )
    parser.add_argument(
        "--custom_system_prompt",
        type=str,
        default=None,
        help="Custom system prompt for the model",
    )
    parser.add_argument(
        "--select_pages", type=str, default=None, help="Pages to select"
    )
    parser.add_argument(
        "--kwargs",
        type=json.loads,
        default="{}",
        help="Additional keyword arguments as JSON string",
    )
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent
    environ.Env.read_env(BASE_DIR / ".env")

    if args.output_dir is None:
        output_dir = pathlib.Path(args.file_path).parent / "markdown_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(output_dir)

    result = asyncio.run(
        to_markdown(
            file_path=args.file_path,
            output_dir=args.output_dir,
            model=args.model,
            custom_system_prompt=args.custom_system_prompt,
            select_pages=args.select_pages,
            kwargs=args.kwargs,
        )
    )

    # uv run python src/article_to_markdown.py --file_path "/home/benjamin.cretois/Code/social-science-AI/test/article.pdf" --model gemini/gemini-2.0-flash
