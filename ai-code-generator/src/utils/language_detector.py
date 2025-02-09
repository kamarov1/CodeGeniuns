import re
import logging
from pathlib import Path
from typing import Optional, Dict, List
from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.util import ClassNotFound
from ..config.settings import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

class LanguageDetector:
    def __init__(self):
        self.file_signatures = self._load_language_signatures()
        self.keyword_patterns = self._load_keyword_patterns()

    def detect(self, input_data: str, file_path: Optional[str] = None) -> str:
        """
        Detecta a linguagem de programação usando múltiplas estratégias
        
        Args:
            input_data (str): Conteúdo do arquivo ou código
            file_path (str, optional): Caminho do arquivo (quando disponível)
            
        Returns:
            str: Linguagem detectada (ex: 'python', 'javascript', etc.)
        """
        detection_methods = [
            self._detect_by_extension,
            self._detect_by_shebang,
            self._detect_by_keywords,
            self._detect_by_pygments,
            self._detect_by_file_signature
        ]

        for method in detection_methods:
            lang = method(input_data, file_path)
            if lang in SUPPORTED_LANGUAGES:
                return lang

        return 'unknown'

    def _detect_by_extension(self, _: str, file_path: Optional[str]) -> Optional[str]:
        """Detecta a linguagem pela extensão do arquivo"""
        if file_path:
            try:
                lexer = get_lexer_for_filename(file_path)
                return lexer.name.lower()
            except ClassNotFound:
                pass
        return None

    def _detect_by_shebang(self, input_data: str, _: Optional[str]) -> Optional[str]:
        """Detecta a linguagem pela linha shebang"""
        shebang_map = {
            'python': ['python', 'python3'],
            'javascript': ['node'],
            'ruby': ['ruby'],
            'perl': ['perl']
        }

        if input_data.startswith('#!'):
            first_line = input_data.split('\n')[0].lower()
            for lang, commands in shebang_map.items():
                if any(cmd in first_line for cmd in commands):
                    return lang
        return None

    def _detect_by_keywords(self, input_data: str, _: Optional[str]) -> Optional[str]:
        """Detecta a linguagem por palavras-chave características"""
        sample = input_data[:1000]  # Analisa apenas os primeiros 1000 caracteres
        scores = {lang: 0 for lang in SUPPORTED_LANGUAGES}

        for lang, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sample):
                    scores[lang] += 1

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return None

    def _detect_by_pygments(self, input_data: str, _: Optional[str]) -> Optional[str]:
        """Usa o Pygments para detectar a linguagem pelo conteúdo"""
        try:
            lexer = guess_lexer(input_data)
            return lexer.name.lower()
        except ClassNotFound:
            return None

    def _detect_by_file_signature(self, input_data: str, _: Optional[str]) -> Optional[str]:
        """Detecta por assinaturas específicas de arquivo"""
        for lang, signatures in self.file_signatures.items():
            if any(re.search(pattern, input_data[:100]) for pattern in signatures):
                return lang
        return None

    def _load_language_signatures(self) -> Dict[str, List[str]]:
        """Carrega assinaturas únicas de cada linguagem"""
        return {
            'python': [
                r'^import\s+\w+',
                r'def\s+\w+\(',
                r'class\s+\w+'
            ],
            'javascript': [
                r'function\s+\w+\(',
                r'const\s+\w+',
                r'let\s+\w+',
                r'console\.log'
            ],
            'java': [
                r'public\s+class',
                r'import\s+java\.',
                r'System\.out\.print'
            ],
            'cpp': [
                r'#include\s+<iostream>',
                r'using\s+namespace\s+std',
                r'std::cout'
            ],
            'go': [
                r'package\s+main',
                r'import\s+"fmt"',
                r'func\s+main\('
            ],
            'html': [
                r'<!DOCTYPE html>',
                r'<html>',
                r'<head>'
            ],
            'css': [
                r'^\s*\.\w+',
                r'^\s*#\w+',
                r'@media'
            ]
        }

    def _load_keyword_patterns(self) -> Dict[str, List[str]]:
        """Carrega padrões de palavras-chave por linguagem"""
        return {
            'python': [
                r'\bdef\b',
                r'\bclass\b',
                r'\bimport\b',
                r'\bfrom\b',
                r'\bas\b'
            ],
            'javascript': [
                r'\bfunction\b',
                r'\bconst\b',
                r'\blet\b',
                r'\bvar\b',
                r'=>'
            ],
            'java': [
                r'\bpublic\b',
                r'\bclass\b',
                r'\bstatic\b',
                r'\bvoid\b',
                r'\bnew\b'
            ],
            'cpp': [
                r'#include\b',
                r'\bnamespace\b',
                r'\bstd::\b',
                r'\bcout\b',
                r'\bendl\b'
            ],
            'go': [
                r'\bpackage\b',
                r'\bimport\b',
                r'\bfunc\b',
                r':=',
                r'\bgo\b'
            ],
            'ruby': [
                r'\bdef\b',
                r'\bclass\b',
                r'\bmodule\b',
                r'\bend\b',
                r'puts'
            ]
        }

def detect_language_from_file(file_path: str) -> str:
    """Detecta a linguagem de um arquivo existente"""
    detector = LanguageDetector()
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        return detector.detect(content, file_path)
    except Exception as e:
        logger.error(f"Erro ao detectar linguagem: {str(e)}")
        return 'unknown'

def detect_language_from_code(code: str) -> str:
    """Detecta a linguagem de um trecho de código"""
    detector = LanguageDetector()
    return detector.detect(code)
