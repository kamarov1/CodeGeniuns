import argparse
import logging
from pathlib import Path

from models.codet5_model import CodeT5Model, CodeT5Config
from api.gemini_api import GeminiAPI
from api.github_copilot import GitHubCopilotAPI
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="AI Code Generator")
    parser.add_argument("--model", type=str, choices=['codet5', 'gemini', 'copilot'], default='codet5', help="Model to use for code generation")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for code generation")
    parser.add_argument("--output", type=str, default="generated_code.py", help="Output file for generated code")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info(f"Using model: {args.model}")

    if args.model == 'codet5':
        config = CodeT5Config()
        model = CodeT5Model(config)
        generated_code = model.generate_code(args.prompt)
    elif args.model == 'gemini':
        api = GeminiAPI()
        generated_code = api.generate_code(args.prompt)
    elif args.model == 'copilot':
        api = GitHubCopilotAPI()
        generated_code = api.generate_code(args.prompt, language="python")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    output_path = Path(args.output)
    output_path.write_text(generated_code)
    logger.info(f"Generated code saved to {output_path}")

if __name__ == "__main__":
    main()