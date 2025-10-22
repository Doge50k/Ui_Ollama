# streamlit_app/app.py
import streamlit as st
import ollama
import logging
import logstash_async.handler

# --- CONFIGURAÇÃO DO LOGGER PARA LOGSTACH ---

LOGSTASH_HOST = 'logstash'
LOGSTASH_PORT = 50000

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


# --- INTERFACE DO STREAMLIT ---

# Informa o endereço do servidor Ollama, usando o nome do serviço do docker-compose
client = ollama.Client(host='http://ollama:11434')

# Configuração da página
st.set_page_config(page_title="Gemma 2B Chatbot", layout="wide")
st.title("🤖 Chat com Gemma 2B")

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra as mensagens do histórico na tela
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
if prompt := st.chat_input("Como posso ajudar?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            
            stream = client.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )

            for chunk in stream:
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

            # Envia o log da interação para o Logstash
            log_interaction(prompt, full_response)
        
        except Exception as e:
            st.error(f"Erro ao conectar com o Ollama: {e}")
            full_response = "Desculpe, não consegui me conectar ao modelo."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
