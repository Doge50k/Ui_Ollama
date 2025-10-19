#!/bin/sh
# entrypoint.sh

# A imagem base do Ollama usa Debian/Ubuntu, então temos que usar o gerenciador 'apt-get'
echo "Instalando a dependência 'curl'..."
apt-get update && apt-get install -y curl

# Inicia o servidor Ollama em segundo plano
/bin/ollama serve &

# Captura o ID do processo (PID) do servidor
pid=$!

echo "Aguardando o servidor Ollama ficar online..."

# Espera até que o servidor esteja respondendo
while ! curl -s -o /dev/null http://localhost:11434; do
    sleep 1
done

echo "Servidor Ollama está online. Baixando o modelo gemma:2b..."

# Baixa o modelo especificado
/bin/ollama pull gemma:2b

echo "Modelo baixado com sucesso. O servidor continuará rodando."

# Traz o processo do servidor para o primeiro plano para que o contêiner não morra
wait $pid