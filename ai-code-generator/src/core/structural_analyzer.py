import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import magic
import pygments
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
from ..utils.file_operations import read_file_safely
from ..config.settings import SUPPORTED_LANGUAGES, CODE_STYLE, PROJECT_STRUCTURE_RULES

class StructuralAnalyzer:
    def __init__(self):
        self.mime = magic.Magic(mime=True)
        self.file_signatures = self._load_file_signatures()
        self.cache = {}

    def analyze_project(self, project_path: str) -> Dict:
        """Executa análise completa do projeto"""
        project_path = Path(project_path).resolve()
        
        return {
            "structure": self.get_directory_structure(project_path),
            "languages": self.detect_project_languages(project_path),
            "metrics": self.calculate_code_metrics(project_path),
            "violations": self.find_style_violations(project_path),
            "dependencies": self.detect_dependencies(project_path)
        }

    def get_directory_structure(self, root_path: Path) -> Dict:
        """Gera representação hierárquica do projeto"""
        structure = {"name": root_path.name, "type": "directory", "children": []}
        
        for entry in root_path.iterdir():
            if entry.is_dir():
                structure["children"].append(self.get_directory_structure(entry))
            else:
                structure["children"].append({
                    "name": entry.name,
                    "type": "file",
                    "metadata": self.get_file_metadata(entry)
                })
        return structure

    def get_file_metadata(self, file_path: Path) -> Dict:
        """Extrai metadados detalhados do arquivo"""
        if file_path in self.cache:
            return self.cache[file_path]

        content = read_file_safely(file_path)
        metadata = {
            "size": file_path.stat().st_size,
            "mime_type": self.mime.from_file(file_path),
            "language": self.detect_file_language(file_path, content),
            "checksum": hashlib.sha256(content.encode()).hexdigest(),
            "line_count": len(content.splitlines()),
            "ast": self.generate_ast(content, file_path)
        }

        self.cache[file_path] = metadata
        return metadata

    def detect_file_language(self, file_path: Path, content: str) -> str:
        """Detecta linguagem usando múltiplas estratégias"""
        # 1. Tentativa por extensão
        try:
            lexer = get_lexer_for_filename(file_path.name)
            return lexer.name.lower()
        except ClassNotFound:
            pass

        # 2. Análise de shebang
        if content.startswith("#!"):
            shebang = content.splitlines()[0]
            if "python" in shebang:
                return "python"
            elif "node" in shebang or "js" in shebang:
                return "javascript"

        # 3. Assinatura de arquivo
        signature = content[:128]
        for lang, patterns in self.file_signatures.items():
            if any(pattern in signature for pattern in patterns):
                return lang

        return "unknown"

    def detect_project_languages(self, root_path: Path) -> Dict:
        """Identifica linguagens usadas no projeto"""
        lang_stats = {}
        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                metadata = self.get_file_metadata(file_path)
                lang = metadata["language"]
                lang_stats[lang] = lang_stats.get(lang, 0) + 1
        return lang_stats

    def find_style_violations(self, root_path: Path) -> List[Dict]:
        """Encontra violações do guia de estilo"""
        violations = []
        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                lang = self.get_file_metadata(file_path)["language"]
                if lang in CODE_STYLE:
                    violations.extend(
                        self._check_style_rules(file_path, CODE_STYLE[lang])
                    )
        return violations

    def _check_style_rules(self, file_path: Path, rules: Dict) -> List[Dict]:
        """Verifica regras específicas da linguagem"""
        violations = []
        content = read_file_safely(file_path)
        
        # Verificação de tamanho de linha
        if "max_line_length" in rules:
            for i, line in enumerate(content.splitlines(), 1):
                if len(line) > rules["max_line_length"]:
                    violations.append({
                        "file": str(file_path),
                        "line": i,
                        "rule": "max_line_length",
                        "message": f"Line exceeds {rules['max_line_length']} characters"
                    })

        # Verificação de indentação
        if "indentation" in rules:
            for i, line in enumerate(content.splitlines(), 1):
                if line.startswith(" " * (rules["indentation"] + 1)):
                    violations.append({
                        "file": str(file_path),
                        "line": i,
                        "rule": "indentation",
                        "message": f"Inconsistent indentation, expected {rules['indentation']} spaces"
                    })

        return violations

    def _load_file_signatures(self) -> Dict[str, List[str]]:
        """Carrega assinaturas de arquivo para detecção"""
        return {
            "python": ["import ", "def ", "class "],
            "javascript": ["function ", "const ", "let "],
            "java": ["public class ", "import java."],
            "html": ["<!DOCTYPE html>", "<html>"],
            "css": ["body {", "@media"]
        }

    def generate_ast(self, content: str, file_path: Path) -> Dict:
        """Gera AST simplificado do código"""
        lang = self.detect_file_language(file_path, content)
        try:
            if lang == "python":
                return self._parse_python_ast(content)
            elif lang == "javascript":
                return self._parse_javascript_ast(content)
        except Exception as e:
            return {"error": str(e)}
        return {}

    def _parse_python_ast(self, content: str) -> Dict:
        """Analisa AST para Python"""
        import ast
        tree = ast.parse(content)
        return {
            "imports": [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)],
            "functions": [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)],
            "classes": [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        }

    def _parse_javascript_ast(self, content: str) -> Dict:
        """Analisa AST para JavaScript (usando esprima)"""
        from esprima import parseScript
        ast = parseScript(content)
        return {
            "functions": [node.id.name for node in ast.body if node.type == 'FunctionDeclaration'],
            "variables": [decl.id.name for node in ast.body if node.type == 'VariableDeclaration' for decl in node.declarations]
        }

    def calculate_code_metrics(self, root_path: Path) -> Dict:
        """Calcula métricas de qualidade de código"""
        metrics = {
            "total_files": 0,
            "total_lines": 0,
            "complexity": 0,
            "languages": {},
            "duplications": self.find_duplicate_code(root_path)
        }

        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                metadata = self.get_file_metadata(file_path)
                lang = metadata["language"]
                
                if lang in SUPPORTED_LANGUAGES:
                    metrics["total_files"] += 1
                    metrics["total_lines"] += metadata["line_count"]
                    metrics["languages"][lang] = metrics["languages"].get(lang, 0) + 1

        return metrics

    def find_duplicate_code(self, root_path: Path) -> List[Dict]:
        """Detecta código duplicado usando hashing"""
        hashes = {}
        duplicates = []

        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                content = read_file_safely(file_path)
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                
                if file_hash in hashes:
                    duplicates.append({
                        "files": [str(file_path), hashes[file_hash]],
                        "hash": file_hash
                    })
                else:
                    hashes[file_hash] = str(file_path)

        return duplicates

    def detect_dependencies(self, root_path: Path) -> Dict:
        """Identifica dependências do projeto"""
        dependencies = {}
        for file_path in root_path.rglob("*"):
            if file_path.name in ["requirements.txt", "package.json"]:
                content = read_file_safely(file_path)
                if file_path.name == "requirements.txt":
                    dependencies["python"] = [line.strip() for line in content.splitlines() if line.strip()]
                elif file_path.name == "package.json":
                    import json
                    package = json.loads(content)
                    dependencies["javascript"] = list(package.get("dependencies", {}).keys())
        return dependencies
