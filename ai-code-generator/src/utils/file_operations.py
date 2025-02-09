import os
import hashlib
from pathlib import Path
from typing import Dict, List, Union
import logging
from ..config.settings import MAX_FILE_SIZE, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

def read_file_safely(file_path: Union[str, Path]) -> str:
    """
    Lê o conteúdo de um arquivo de forma segura, com verificações de tamanho e tipo.

    Args:
        file_path (Union[str, Path]): Caminho do arquivo a ser lido.

    Returns:
        str: Conteúdo do arquivo.

    Raises:
        ValueError: Se o arquivo for muito grande ou de um tipo não suportado.
        IOError: Se ocorrer um erro ao ler o arquivo.
    """
    file_path = Path(file_path)
    try:
        if not file_path.is_file():
            raise ValueError(f"O caminho {file_path} não é um arquivo válido.")
        
        if file_path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(f"Arquivo muito grande. Tamanho máximo permitido: {MAX_FILE_SIZE} bytes.")
        
        if file_path.suffix[1:] not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Tipo de arquivo não suportado: {file_path.suffix}")
        
        with file_path.open('r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo {file_path}: {str(e)}")
        raise IOError(f"Não foi possível ler o arquivo {file_path}: {str(e)}")

def write_file_safely(file_path: Union[str, Path], content: str) -> None:
    """
    Escreve conteúdo em um arquivo de forma segura, criando diretórios se necessário.

    Args:
        file_path (Union[str, Path]): Caminho do arquivo a ser escrito.
        content (str): Conteúdo a ser escrito no arquivo.

    Raises:
        IOError: Se ocorrer um erro ao escrever o arquivo.
    """
    file_path = Path(file_path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('w', encoding='utf-8') as file:
            file.write(content)
        logger.info(f"Arquivo escrito com sucesso: {file_path}")
    except Exception as e:
        logger.error(f"Erro ao escrever o arquivo {file_path}: {str(e)}")
        raise IOError(f"Não foi possível escrever o arquivo {file_path}: {str(e)}")

def validate_directory_structure(root_path: Union[str, Path]) -> Dict[str, Union[str, List[str]]]:
    """
    Valida a estrutura de diretórios de um projeto.

    Args:
        root_path (Union[str, Path]): Caminho raiz do projeto.

    Returns:
        Dict[str, Union[str, List[str]]]: Dicionário com informações sobre a estrutura do projeto.
    """
    root_path = Path(root_path)
    structure = {
        "root": str(root_path),
        "directories": [],
        "files": [],
        "issues": []
    }

    try:
        for item in root_path.rglob('*'):
            rel_path = item.relative_to(root_path)
            if item.is_dir():
                structure["directories"].append(str(rel_path))
            elif item.is_file():
                structure["files"].append(str(rel_path))

        # Verificações básicas de estrutura
        if not any(f.startswith('src') for f in structure["directories"]):
            structure["issues"].append("Diretório 'src' não encontrado")
        if not any(f.endswith('.py') for f in structure["files"]):
            structure["issues"].append("Nenhum arquivo Python encontrado")

    except Exception as e:
        logger.error(f"Erro ao validar estrutura de diretórios: {str(e)}")
        structure["issues"].append(f"Erro na validação: {str(e)}")

    return structure

def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calcula o hash SHA-256 de um arquivo.

    Args:
        file_path (Union[str, Path]): Caminho do arquivo.

    Returns:
        str: Hash SHA-256 do arquivo.

    Raises:
        IOError: Se ocorrer um erro ao ler o arquivo.
    """
    file_path = Path(file_path)
    try:
        with file_path.open('rb') as file:
            file_hash = hashlib.sha256()
            chunk = file.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = file.read(8192)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"Erro ao calcular hash do arquivo {file_path}: {str(e)}")
        raise IOError(f"Não foi possível calcular o hash do arquivo {file_path}: {str(e)}")

def find_duplicate_files(directory: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Encontra arquivos duplicados em um diretório com base no hash do conteúdo.

    Args:
        directory (Union[str, Path]): Diretório a ser analisado.

    Returns:
        Dict[str, List[str]]: Dicionário com hashes como chaves e listas de caminhos de arquivos duplicados como valores.
    """
    directory = Path(directory)
    hash_dict = {}

    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_hash = calculate_file_hash(file_path)
                hash_dict.setdefault(file_hash, []).append(str(file_path))

        # Filtra apenas os hashes com mais de um arquivo
        return {h: files for h, files in hash_dict.items() if len(files) > 1}
    except Exception as e:
        logger.error(f"Erro ao procurar arquivos duplicados: {str(e)}")
        return {}

def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Union[str, int]]:
    """
    Obtém metadados de um arquivo.

    Args:
        file_path (Union[str, Path]): Caminho do arquivo.

    Returns:
        Dict[str, Union[str, int]]: Dicionário com metadados do arquivo.
    """
    file_path = Path(file_path)
    try:
        stats = file_path.stat()
        return {
            "name": file_path.name,
            "size": stats.st_size,
            "created": stats.st_ctime,
            "modified": stats.st_mtime,
            "extension": file_path.suffix,
            "hash": calculate_file_hash(file_path)
        }
    except Exception as e:
        logger.error(f"Erro ao obter metadados do arquivo {file_path}: {str(e)}")
        return {}
