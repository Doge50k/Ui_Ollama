import streamlit as st 
import logging
import logstash_async.handler
import os

# Importações do LangChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore

# --- CONFIGURAÇÃO DO LOGGER PARA LOGSTACH ---

LOGSTASH_HOST = 'logstash' # Nome do serviço do Logstash no docker-compose
LOGSTASH_PORT = 50000 # Porta configurada no Logstash para receber logs

# Cria o logger
logger = logging.getLogger('logstash_logger')
logger.setLevel(logging.INFO)

# Garante que o logger seja criado apenas uma vez
if not logger.handlers:
    # Cria o handler que envia logs via TCP para o Logstash
    handler = logstash_async.handler.AsynchronousLogstashHandler(
        host=LOGSTASH_HOST,
        port=LOGSTASH_PORT,
        database_path=None  # Arquivo para armazenar logs não enviados
    )

    # Adiciona o handler ao logger
    logger.addHandler(handler)

def log_interaction(prompt, response):
    """Formata a interação como um dicionário e envia para o Logstash."""
    log_entry = {
        'prompt': prompt,
        'response': response
    }
    # O handler irá converter o dicionário em JSON e enviar para o Logstash
    logger.info('Chat Interaction', extra=log_entry)

# --- CONFIGURAÇÕES DO RAG E OLLAMA ---
OLLAMA_HOST = 'http://ollama:11434'
ELASTIC_URL = "http://elasticsearch:9200"
INDEX_NAME = "documentos_rag"

# Conecta ao Ollama
llm = ChatOllama(model="gemma:2b", base_url=OLLAMA_HOST)

# Prepara o armazenamento vetorial para busca (retriever)
embeddings = OllamaEmbeddings(model="gemma:2b", base_url=OLLAMA_HOST)
vector_store = ElasticsearchStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    es_url=ELASTIC_URL
)
retriever = vector_store.as_retriever()

# Template do prompt para guiar o modelo no uso do contexto

template = """
Você é um assistente de chatbot. Responda à perguntas do usuário baseando-se apenas no contexto fornecido.
Se a resposta não estiver no contexto, diga que você não têm a informação.
Seja conciso e direto.

Contexto:
{context}

Pergunta: 
{question}


Responda em Português (PT-BR).
Resposta:
"""
prompt_template = PromptTemplate.from_template(template)

# Cria a cadeia RAG com LangChain
generation_chain = (
    prompt_template
    | llm
    | StrOutputParser()
)

def format_docs(docs):
    """Prepara os documentos recuperados para serem inseridos no prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- INTERFACE DO STREAMLIT ---

# Configuração da página
st.set_page_config(page_title="Gemma 2B Chatbot com RAG", layout="wide")
st.title("🤖 Chat com Gemma 2B")

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra as mensagens do histórico na tela
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Fontes Consultadas"):
                for source in message["sources"]:
                    st.info(f"Arquivo: {source}")

# Entrada das mensagens
if prompt := st.chat_input("Pergunte sobre seus documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        source_files = [] # Reseta as fontes
        
        try:
            # Recupera os documentos relevantes
            retrieved_docs = retriever.invoke(prompt)
            
            # Extrai os nomes dos arquivos como fontes
            if retrieved_docs:
                source_files = list(set([os.path.basename(doc.metadata.get("source", "Desconhecido")) for doc in retrieved_docs]))
            
            # Formata o contexto para o prompt
            formatted_context = format_docs(retrieved_docs)
            
            # Monta o dicionário que a cadeira de geração espera
            chain_input = {
                "context": formatted_context,
                "question": prompt
            }

            # Gera a resposta com a cadeia RAG
            response_stream = generation_chain.stream(chain_input)

            # Exibe a resposta em streaming
            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Exibe as fontes logo abaixo da resposta
            if source_files:
                with st.expander("Fontes Consultadas"):
                    for source in source_files:
                        st.info(f"Arquivo: {source}")

            log_interaction(prompt, full_response)
        
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            full_response = "Desculpe, não consegui processar sua solicitação."

    # Salva a resposta e as fontes no histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": source_files
    })