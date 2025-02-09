# src/api/github_copilot.py

import openai
from typing import List, Dict, Any, Optional
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GitHubCopilotAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables.")
        
        openai.api_key = self.api_key
        logger.info("GitHub Copilot API (simulated) initialized successfully.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_code(self, prompt: str, language: str, **kwargs) -> str:
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",  # Using a more recent model
                prompt=f"# Language: {language}\n# Task: {prompt}\n\n",
                max_tokens=kwargs.get('max_tokens', 150),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0),
                presence_penalty=kwargs.get('presence_penalty', 0),
                stop=["# Language:", "# Task:"]
            )

            generated_code = response.choices[0].text.strip()
            return generated_code

        except Exception as e:
            logger.error(f"Error generating code with GitHub Copilot API: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_completions(self, code_context: str, cursor_position: int, language: str, **kwargs) -> List[str]:
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=f"# Language: {language}\n{code_context[:cursor_position]}",
                max_tokens=kwargs.get('max_tokens', 50),
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 1.0),
                n=kwargs.get('n', 3),  # Number of completions to generate
                stop=["\n\n"]
            )

            completions = [choice.text.strip() for choice in response.choices]
            return completions

        except Exception as e:
            logger.error(f"Error getting completions from GitHub Copilot API: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def explain_code(self, code: str, language: str) -> str:
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=f"# Language: {language}\n# Explain the following code:\n{code}\n\nExplanation:",
                max_tokens=200,
                temperature=0.5,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0
            )

            explanation = response.choices[0].text.strip()
            return explanation

        except Exception as e:
            logger.error(f"Error explaining code with GitHub Copilot API: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        # Note: This is simulated information as the actual Copilot model details aren't public
        return {
            "name": "GitHub Copilot (simulated)",
            "version": "1.0",
            "description": "AI-powered code completion tool",
            "max_tokens": 150,
            "supported_languages": ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "PHP", "C#", "TypeScript", "SQL"],
        }

    def __repr__(self) -> str:
        return "GitHubCopilotAPI(simulated)"

