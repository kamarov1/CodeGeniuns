# setup_environment.ps1

# Verifica se o Python está instalado
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python não está instalado. Por favor, instale o Python e tente novamente."
    exit 1
}

# Cria e ativa o ambiente virtual
python -m venv venv
.\venv\Scripts\Activate

# Atualiza pip
python -m pip install --upgrade pip

# Instala as dependências
pip install torch transformers datasets tqdm pandas numpy scikit-learn matplotlib seaborn

# Configura as variáveis de ambiente
$env:PYTHONPATH = "$PWD"
$env:TOKENIZERS_PARALLELISM = "false"

# Verifica se as chaves de API estão configuradas
if (-not $env:OPENAI_API_KEY) {
    Write-Warning "A variável de ambiente OPENAI_API_KEY não está configurada. Por favor, configure-a antes de usar a API do OpenAI."
}

if (-not $env:GEMINI_API_KEY) {
    Write-Warning "A variável de ambiente GEMINI_API_KEY não está configurada. Por favor, configure-a antes de usar a API do Gemini."
}

Write-Host "Ambiente configurado com sucesso! Use 'deactivate' para sair do ambiente virtual quando terminar."
