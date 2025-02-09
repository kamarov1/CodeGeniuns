import os
from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Configurações do NLP
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1000"))

# Configurações do CodeGenerator
SUPPORTED_LANGUAGES = ["python", "javascript", "java", "cpp", "go"]

# Configurações de cache
CACHE_DIR = BASE_DIR / "cache"
CACHE_EXPIRATION = int(os.environ.get("CACHE_EXPIRATION", "3600"))  # em segundos

# Configurações de logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "ai_code_generator.log"

# Configurações de segurança
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", "60"))
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Configurações de performance
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

# Configurações de autoaprendizagem
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
EVALUATION_INTERVAL = int(os.environ.get("EVALUATION_INTERVAL", "1000"))

# Configurações de output
OUTPUT_DIR = BASE_DIR / "output"
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", str(1024 * 1024)))  # 1MB

# Configurações de teste
TEST_DATA_DIR = BASE_DIR / "tests" / "data"

# Configurações de plugins
PLUGIN_DIR = BASE_DIR / "src" / "plugins"
ENABLED_PLUGINS = os.environ.get("ENABLED_PLUGINS", "").split(",")

# Configurações de estilo de código
CODE_STYLE = {
    "python": {
        "indentation": 4,
        "max_line_length": 88,
        "quote_type": "double"
    },
    "javascript": {
        "indentation": 2,
        "semicolons": True,
        "quote_type": "single"
    }
}

# Configurações de documentação
GENERATE_DOCS = os.environ.get("GENERATE_DOCS", "True").lower() == "true"
DOCS_OUTPUT_DIR = BASE_DIR / "docs" / "generated"

# Configurações de versionamento
VERSION = "1.0.0"
REQUIRE_VERSION_HEADER = os.environ.get("REQUIRE_VERSION_HEADER", "False").lower() == "true"

# Configurações de feedback e erro
MAX_ERROR_REPORTS = int(os.environ.get("MAX_ERROR_REPORTS", "100"))
FEEDBACK_EMAIL = os.environ.get("FEEDBACK_EMAIL", "support@aicodegenerator.com")

# Configurações de ambiente
ENV = os.environ.get("ENV", "development")
DEBUG = ENV == "development"

# Configurações específicas do modelo de aprendizado
MODEL_SAVE_PATH = BASE_DIR / "models"
FEEDBACK_THRESHOLD = int(os.environ.get("FEEDBACK_THRESHOLD", "100"))
RETRAINING_INTERVAL = int(os.environ.get("RETRAINING_INTERVAL", "7"))  # em dias

# Inicialização de diretórios
for directory in [CACHE_DIR, OUTPUT_DIR, LOG_FILE.parent, DOCS_OUTPUT_DIR, MODEL_SAVE_PATH]:
    directory.mkdir(parents=True, exist_ok=True)
