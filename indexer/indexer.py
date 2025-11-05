# indexer/indexer.py
import os
import time
import logging
import requests
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
import ollama

# --- Configurações ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ELASTIC_URL = "http://elasticsearch:9200"
OLLAMA_URL = "http://ollama:11434"
INDEX_NAME = "documentos_rag"
DOCS_SOURCE_DIRECTORY = "/app/docs_augmentation" # Caminho dentro do contêiner

def wait_for_service(url, service_name, check_func=None):
    """Espera um serviço ficar online fazendo requisições GET."""
    logging.info(f"Aguardando o serviço: {service_name} em {url}")
    while True:
        try:
            if check_func:
                check_func()
            else:
                requests.get(url, timeout=5)
            logging.info(f"Serviço {service_name} está online.")
            break
        except Exception as e:
            logging.warning(f"Serviço {service_name} ainda não está pronto. Tentando novamente em 15s...")
            time.sleep(15)

def check_elastic_connection():
    """Verifica a conexão com o Elasticsearch."""
    es = Elasticsearch(ELASTIC_URL)
    if not es.ping():
        raise ConnectionError("Ping do Elasticsearch falhou")

def check_ollama_connection():
    """Verifica a conexão com o Ollama."""
    ollama.Client(host=OLLAMA_URL).list()

def get_local_files():
    """Varre o diretório de documentos e retorna uma lista de arquivos."""
    local_files = []
    for root, _, files in os.walk(DOCS_SOURCE_DIRECTORY):
        for file in files:
            if file.endswith('.txt'):
                local_files.append(os.path.join(root, file))
    return local_files

def get_indexed_files(es_client):
    """Consulta o Elasticsearch para obter a lista de arquivos já indexados."""
    indexed_files = set()
    try:
        #Verifica se o índice existe antes de consultá-lo
        if not es_client.indices.exists(index=INDEX_NAME):
            logging.info(f"O índice '{INDEX_NAME}' ainda não existe. Nenhum documento indexado.")
            return indexed_files
        
        # Query de agregação para buscar todos os valores únicos de campo 'metadata.source'
        query = {
            "size": 0,
            "aggs": {
                "distinct_sources": {
                    "terms": {
                        "field": "metadata.source.keyword",
                        "size": 10000  # Ajuste conforme necessário
                    }
                }
            }
        }
        response = es_client.search(index=INDEX_NAME, body=query)

        # Extrai os nomes dos arquivos dos "buckets" da agregação
        buckets = response.get('aggregations', {}).get('distinct_sources', {}).get('buckets', [])
        for bucket in buckets:
            indexed_files.add(bucket['key'])
        
        logging.info(f"Encontrados {len(indexed_files)} arquivos indexados no Elasticsearch.")
        
    except Exception as e:
        logging.error(f"Erro ao consultar arquivos indexados no Elasticsearch: {e}")
        return indexed_files


def main():
    """
    Função principal que carrega, processa e indexa os documentos.
    """
    # Espera os serviços dependerem
    wait_for_service(ELASTIC_URL, "Elasticsearch", check_elastic_connection)
    wait_for_service(OLLAMA_URL, "Ollama", check_ollama_connection)

    # Verifica se a indexação já foi feita
    es_client = Elasticsearch(ELASTIC_URL)

    # Compara arquivos locais com os já indexados
    local_files = get_local_files()
    indexed_files = get_indexed_files(es_client)

    # Usa 'set' para encontrar a diferença facilmente
    files_to_index = [f for f in local_files if f not in indexed_files]
    
    if not files_to_index:
        logging.info("Todos os documentos já estão indexados. Nenhum novo documento para indexar.")
        return
    
    logging.info(f"Encontrados {len(files_to_index)} novos documentos para indexar.")

    # Carrega e processa apenas os novos documentos
    all_new_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file_path in files_to_index:
        try:
            logging.info(f"Processando o arquivo: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            docs_split = text_splitter.split_documents(docs)
            all_new_docs.extend(docs_split)
        except Exception as e:
            logging.error(f"Erro ao processar o arquivo {file_path}: {e}")


    if not all_new_docs:
        logging.info("Nenhum novo documento foi processado. Encerrando.")
        return
    
    # Cria embbeddings e indexa no Elasticsearch
    logging.info("Criando embeddings e indexando documentos no Elasticsearch...")
    embeddings = OllamaEmbeddings(
        model="gemma:2b",
        base_url=OLLAMA_URL
    )

    logging.info(f"Adicionando {len(all_new_docs)} novos pedaços ao Elasticsearch...")
    ElasticsearchStore.from_documents(
        all_new_docs,
        embeddings,
        es_url=ELASTIC_URL,
        index_name=INDEX_NAME
    )
    logging.info("Indexação concluída com sucesso.")


if __name__ == "__main__":
    main()
