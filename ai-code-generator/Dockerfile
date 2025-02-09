# Use uma imagem base oficial do Python
FROM python:3.9-slim-buster

# Define o diretório de trabalho no container
WORKDIR /app

# Copia os arquivos de requisitos para o diretório de trabalho
COPY requirements.txt .

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto para o diretório de trabalho
COPY . .

# Configura as variáveis de ambiente
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

# Expõe a porta que a aplicação irá rodar (ajuste conforme necessário)
EXPOSE 8000

# Define o comando para rodar a aplicação
CMD ["python", "src/main.py"]
