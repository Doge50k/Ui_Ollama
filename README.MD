
# Guia de Execução: Rodando Ollama/gemma:2b localmente em UI interativa

Este projeto fornece um ambiente fácil de executar para rodar um chatbot localmente usando Docker. A interface web é construída com **Streamlit** e se conecta a um modelo de linguagem leve (**gemma:2b**) servido pelo **Ollama**.

**Aviso:** O guia foi testado e validado em um ambiente **Linux**. Para outros sistemas operacionais como Windows ou macOS, podem ser necessárias pequenas adaptações (principalmente no Passo 2).

### ## Pré-requisitos

# Antes de começar, garanta que você tenha os seguintes softwares instalados:
# - Docker
# - Docker Compose (geralmente incluído no Docker Desktop)

### ## Passo 1: Estrutura de Arquivos

# Certifique-se de que a estrutura do seu projeto está organizada da seguinte forma:

# llm-gemma-ollama/
# ├── docker-compose.yml
# ├── ollama_config/
# │   └── entrypoint.sh
# └── streamlit_app/
#     ├── Dockerfile
#     ├── requirements.txt
#     └── app.py

### ## Passo 2: Tornar o Script de Inicialização Executável

# No linux o Docker precisa de permissão para executar o script que automatiza o download do modelo.
# Abra um terminal na pasta raiz do seu projeto e execute o comando abaixo:

```bash
chmod +x ./ollama_config/entrypoint.sh
```

### ## Passo 3: Construir e Iniciar os Contêineres

# Este comando deve ser executado na pasta raiz do projeto.

```bash
docker-compose up -d --build
```

# Na primeira vez, este passo pode demorar alguns minutos, pois o Ollama irá baixar o modelo gemma:2b (aprox. 1.7GB).
# Você pode acompanhar o progresso do download com o seguinte comando em um novo terminal:

```bash
docker logs ollama_server -f
```

# Aguarde até que a mensagem "Modelo baixado com sucesso" apareça nos logs.

### ## Passo 4: Acessar o Chatbot

# Após os contêineres estarem rodando e o modelo ter sido baixado, a aplicação estará pronta.

# Abra o seu navegador de internet e acesse o seguinte endereço:

[http://localhost:8501](http://localhost:8501)

### ## Passo 5: Como Parar a Aplicação

# Execute o seguinte comando na pasta raiz do projeto caso queira parar e remover os contêineres:

```bash
docker-compose down
```

# Os dados do modelo baixado pelo Ollama serão preservados em um volume, então você não precisará baixá-lo novamente na próxima vez que iniciar o projeto.

### ## Solução de Problemas Comuns

* Erro de Permissão no Docker (Linux): Se você receber um erro como permission denied while trying to connect to the Docker daemon, pode ser necessário rodar os comandos Docker com sudo ou configurar o [Docker para rodar como um usuário não-root](https://docs.docker.com/engine/install/linux-postinstall/)

* Erro de Conexão no Streamlit: Se a interface mostrar um erro de Connection refused, verifique os logs do contêiner ollama_server com docker logs ollama_server. Certifique-se de que ele iniciou corretamente.