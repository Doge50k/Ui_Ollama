
# Guia de Execução: Rodando Ollama/gemma:2b localmente em UI interativa e monitoramento com ELK

Este projeto fornece um ambiente fácil de executar para rodar um chatbot localmente usando Docker. A interface web é construída com **Streamlit** e se conecta a um modelo de linguagem leve (**gemma:2b**) servido pelo **Ollama**. Além disso, o projeto integra a stack **Elastic (ELK)** para coleta, armazenamento e visualização dos logs de interação do chatbot.

**Aviso:** O guia foi testado e validado em um ambiente **Linux**. Para outros sistemas operacionais como Windows ou macOS, podem ser necessárias pequenas adaptações (principalmente no Passo 2).

### Arquitetura

O projeto é composto pelos seguintes serviços, que se comunicam através de uma rede Docker:

1.  **Streamlit (`streamlit-app`):** O frontend da aplicação, onde o usuário interage com o chatbot. Ele envia os prompts para o Ollama e envia os logs de interação para o Logstash.
2.  **Ollama (`ollama`):** O backend que serve o modelo de linguagem `gemma:2b`.
3.  **Logstash (`logstash`):** Recebe os logs da aplicação Streamlit via TCP, processa-os e os envia para o Elasticsearch.
4.  **Elasticsearch (`elasticsearch`):** Um banco de dados NoSQL que armazena e indexa os logs de forma eficiente.
5.  **Kibana (`kibana`):** Uma ferramenta de visualização que se conecta ao Elasticsearch, permitindo a criação de dashboards e a exploração dos dados.

### Pré-requisitos

### Antes de começar, garanta que você tenha os seguintes softwares instalados:
* Docker
* Docker Compose (geralmente incluído no Docker Desktop)

### Passo 1: Estrutura de Arquivos

Certifique-se de que a estrutura do seu projeto está organizada da seguinte forma:

```bash
llm-gemma-ollama/
├── docker-compose.yml
├── logstash/
│   └── Logstash.conf
├── ollama_config/
│   └── entrypoint.sh
└── streamlit_app/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py
```

### Passo 2: Tornar o Script de Inicialização Executável

No linux o Docker precisa de permissão para executar o script que automatiza o download do modelo.

Abra um terminal na pasta raiz do seu projeto e execute o comando abaixo:

```bash
chmod +x ./ollama_config/entrypoint.sh
```

### Passo 3: Construir e Iniciar os Contêineres

Este comando deve ser executado na pasta raiz do projeto.

```bash
docker-compose up -d --build
```

Na primeira vez, este passo pode demorar alguns minutos, pois o Ollama irá baixar o modelo gemma:2b (aprox. 1.7GB).
Você pode acompanhar o progresso do download com o seguinte comando em um novo terminal:

```bash
docker logs ollama_server -f
```

Aguarde até que a mensagem "Modelo baixado com sucesso" apareça nos logs.

### Passo 4: Acessar os serviços

Após os contêineres estarem rodando e o modelo ter sido baixado, a aplicação estará pronta.

* Para acessar o Chatbot: Abra o seu navegador e acesse:[http://localhost:8501](http://localhost:8501)
* Para acessar o Kibana e visualizar os logs: Abra o seu navegador e acesse: [http://localhost:5601](http://localhost:5601)

## Passo 5: Visualizando os Logs no Kibana 8.19.5

Ao acessar o Kibana pela primeira vez, você precisará configurar o índice para visualizar os logs.

1. No menu lateral, vá em **Management > Stack Management**
2. Clique em **Data Views**(dentro de Kibana) e depois em **Create data view**
3. No campo **Index pattern**, digite ```streamlit-logs-*```.
4. Selecione um campo de timestamp (por exemplo, ```@timestamp```) e clique em **Save data view to Kibana**
5. Agora, no menu lateral principal, vá em **Analytics > Discover** para ver os logs das interações com o chatbot.


## Passo 6: Como Parar a Aplicação

Execute o seguinte comando na pasta raiz do projeto caso queira parar e remover os contêineres:

```bash
docker-compose down
```

Os dados do modelo baixado pelo Ollama e os dados do Elasticsearch serão preservados em volumes, então você não precisará baixá-lo novamente na próxima vez que iniciar o projeto.
Para remover os volumes e começar do zero, execute ```docker-compose down -v```

## Solução de Problemas Comuns

* **Erro de Permissão no Docker (Linux):** Se você receber um erro como ```permission denied while trying to connect to the Docker daemon```, pode ser necessário rodar os comandos Docker com ```sudo``` ou configurar o [Docker para rodar como um usuário não-root](https://docs.docker.com/engine/install/linux-postinstall/)

* **Erro de Conexão no Streamlit:** Se a interface mostrar um erro de ```Connection refused```, verifique os logs do contêiner ```ollama_server``` com ```docker logs ollama_server```. Certifique-se de que ele iniciou corretamente.

* **Kibana não conecta ao Elasticsearch:** Verifique os logs do Kibana com ```docker logs kibana_dashboard```. O problema mais comum é o Elasticsearch não ter iniciado completamente antes do kibana. Uma reinicialização do contêiner do kibana (```docker restart kibana_dashboard```) geralmente resolve o problema.

* **Nenhum Log no Kibana:** Verifique se o Streamlit está rodando, se você gerou logs interagindo com o chatbot, e se os logs do Logstash (```docker logs logstash_processor```) não mostram erros de conexão com o Elasticsearch. Certifique-se também de que o filtro de tempo no Kibana está abrangendo o período em que os logs foram gerados.