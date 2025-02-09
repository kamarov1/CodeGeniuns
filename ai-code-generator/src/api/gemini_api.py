# src/api/gemini_api.py

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, Content, Part
from google.generativeai.types.model import Model
from typing import List, Dict, Any, Optional, Union
import os
import logging
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GeminiAPI:
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-pro'):
        self.api_key = api_key or os.environ.get("AIzaSyDh8DmjiwqDfu0pTfe5otGj8TvX_KBNcCs")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment variables.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.embedding_model = genai.GenerativeModel('embedding-001')
        
        logger.info(f"Gemini API initialized successfully with model: {model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_code(self, prompt: str, **kwargs) -> str:
        try:
            generation_config = self._create_generation_config(**kwargs)

            response = self.model.generate_content(
                Content(parts=[Part.from_text(prompt)]),
                generation_config=generation_config
            )

            if response.text:
                return response.text
            else:
                logger.warning("Gemini API returned an empty response.")
                return ""

        except Exception as e:
            logger.error(f"Error generating code with Gemini API: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            for message in messages[:-1]:  # Process all messages except the last one
                role = message['role']
                content = message['content']
                if role == 'user':
                    chat.send_message(content)
                elif role == 'assistant':
                    chat.history.append({"role": "model", "parts": [content]})

            generation_config = self._create_generation_config(**kwargs)

            response = chat.send_message(
                messages[-1]['content'],
                generation_config=generation_config
            )

            if response.text:
                return response.text
            else:
                logger.warning("Gemini API returned an empty response in chat.")
                return ""

        except Exception as e:
            logger.error(f"Error in chat with Gemini API: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            if isinstance(text, str):
                embedding = self.embedding_model.embed_content(text)
                return embedding.values
            elif isinstance(text, list):
                embeddings = [self.embedding_model.embed_content(t).values for t in text]
                return embeddings
            else:
                raise ValueError("Input must be a string or a list of strings")
        except Exception as e:
            logger.error(f"Error generating embedding with Gemini API: {str(e)}")
            raise

    @lru_cache(maxsize=1)
    def get_model_info(self) -> Dict[str, Any]:
        try:
            model_info = self.model.list_models()[0]
            return self._model_to_dict(model_info)
        except Exception as e:
            logger.error(f"Error retrieving model info from Gemini API: {str(e)}")
            raise

    @staticmethod
    def _create_generation_config(**kwargs) -> GenerationConfig:
        return GenerationConfig(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.95),
            top_k=kwargs.get('top_k', 40),
            max_output_tokens=kwargs.get('max_output_tokens', 1024),
        )

    @staticmethod
    def _model_to_dict(model: Model) -> Dict[str, Any]:
        return {
            "name": model.name,
            "version": model.version,
            "display_name": model.display_name,
            "description": model.description,
            "input_token_limit": model.input_token_limit,
            "output_token_limit": model.output_token_limit,
            "supported_generation_methods": model.supported_generation_methods,
        }

    def __repr__(self) -> str:
        return f"GeminiAPI(model={self.model.model_name})"
