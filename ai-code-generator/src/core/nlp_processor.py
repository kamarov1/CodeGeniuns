import spacy
from typing import Dict, List, Union
import re
from ..utils.language_detector import detect_programming_language
from ..config.settings import SPACY_MODEL, MAX_LENGTH

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load(SPACY_MODEL)
        self.code_patterns = self._compile_code_patterns()

    def _compile_code_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "function": re.compile(r"def\s+(\w+)\s*\((.*?)\)"),
            "class": re.compile(r"class\s+(\w+)"),
            "variable": re.compile(r"(\w+)\s*=\s*(.+)")
        }

    def parse_prompt(self, prompt: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Processa o prompt do usuário e extrai informações relevantes.

        Args:
            prompt (str): O prompt do usuário.

        Returns:
            Dict[str, Union[str, List[Dict[str, str]]]]: Dicionário contendo informações extraídas.
        """
        doc = self.nlp(prompt[:MAX_LENGTH])
        
        extracted_info = {
            "intent": self._extract_intent(doc),
            "language": detect_programming_language(prompt),
            "components": self._extract_components(doc),
            "constraints": self._extract_constraints(doc),
            "context": self._extract_context(prompt)
        }
        
        return self._enrich_extracted_info(extracted_info, doc)

    def _extract_intent(self, doc) -> str:
        """Extrai a intenção principal do prompt."""
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.lemma_
        return "generate"  # Default intent

    def _extract_components(self, doc) -> List[Dict[str, str]]:
        """Extrai componentes de código mencionados no prompt."""
        components = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                components.append({"type": "class", "name": ent.text})
            elif ent.label_ == "PERSON":
                components.append({"type": "function", "name": ent.text})
        return components

    def _extract_constraints(self, doc) -> List[str]:
        """Extrai restrições ou requisitos específicos mencionados no prompt."""
        constraints = []
        for sent in doc.sents:
            if any(token.text.lower() in ["must", "should", "need", "require"] for token in sent):
                constraints.append(sent.text)
        return constraints

    def _extract_context(self, prompt: str) -> Dict[str, List[str]]:
        """Extrai snippets de código ou contexto técnico do prompt."""
        context = {}
        for pattern_name, pattern in self.code_patterns.items():
            matches = pattern.findall(prompt)
            if matches:
                context[pattern_name] = matches
        return context

    def _enrich_extracted_info(self, info: Dict, doc) -> Dict:
        """Enriquece as informações extraídas usando análise linguística avançada."""
        info["key_phrases"] = self._extract_key_phrases(doc)
        info["dependencies"] = self._extract_dependencies(doc)
        info["sentiment"] = self._analyze_sentiment(doc)
        return info

    def _extract_key_phrases(self, doc) -> List[str]:
        """Extrai frases-chave do documento."""
        return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

    def _extract_dependencies(self, doc) -> List[Dict[str, str]]:
        """Extrai relações de dependência relevantes."""
        return [{"word": token.text, "dep": token.dep_, "head": token.head.text}
                for token in doc if token.dep_ in ["nsubj", "dobj", "pobj"]]

    def _analyze_sentiment(self, doc) -> Dict[str, float]:
        """Analisa o sentimento geral do prompt."""
        return {"polarity": doc.sentiment}

    def generate_code_structure(self, parsed_info: Dict) -> str:
        """
        Gera uma estrutura de código base com base nas informações extraídas.

        Args:
            parsed_info (Dict): Informações extraídas do prompt.

        Returns:
            str: Estrutura de código base.
        """
        language = parsed_info["language"]
        components = parsed_info["components"]
        
        if language == "python":
            return self._generate_python_structure(components)
        elif language == "javascript":
            return self._generate_javascript_structure(components)
        else:
            return f"# Code structure for {language}\n# Add implementation here"

    def _generate_python_structure(self, components: List[Dict[str, str]]) -> str:
        structure = ""
        for component in components:
            if component["type"] == "class":
                structure += f"class {component['name']}:\n    def __init__(self):\n        pass\n\n"
            elif component["type"] == "function":
                structure += f"def {component['name']}():\n    pass\n\n"
        return structure

    def _generate_javascript_structure(self, components: List[Dict[str, str]]) -> str:
        structure = ""
        for component in components:
            if component["type"] == "class":
                structure += f"class {component['name']} {{\n    constructor() {{\n    }}\n}}\n\n"
            elif component["type"] == "function":
                structure += f"function {component['name']}() {{\n}}\n\n"
        return structure
