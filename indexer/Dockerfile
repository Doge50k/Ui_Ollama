FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script de indexação e os documentos para o contêiner
COPY indexer.py .
# Os Documentos serão montados via volume, mas copiar garante que a pasta exista
RUN mkdir -p /app/docs_augmentation

# Comando que será executado ao iniciar o contêiner
CMD ["python","indexer.py"]