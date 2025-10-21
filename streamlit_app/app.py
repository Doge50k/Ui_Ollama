# streamlit_app/app.py
import streamlit as st
import ollama

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
            # Usa o 'client' configurado em vez do 'ollama' padrão
            stream = client.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )

            for chunk in stream:
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"Erro ao conectar com o Ollama: {e}")
            full_response = "Desculpe, não consegui me conectar ao modelo."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
