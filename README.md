# 🎓 Chatbot EBP — UNIFEI

Chatbot com inteligência artificial para suporte às coordenações do curso de Engenharia de Bioprocessos (EBP) da Universidade Federal de Itajubá (UNIFEI).

Utiliza **Gemini 2.5 Flash** (Google) via API, orquestrado pela **LangChain**, com **RAG** (Retrieval-Augmented Generation) e **ChromaDB** como banco vetorial.

## Requisitos

- Python 3.10+
- Chave de API do Google Gemini ([obter aqui](https://aistudio.google.com/apikey))

## Instalação

### 1. Extraia o zip e acesse a pasta

```bash
# Extraia o chatbot-ebp.zip e entre na pasta
cd chatbot-ebp
```

### 2. Crie e ative o ambiente virtual

```bash
# Criar
python -m venv venv

# Ativar (Windows PowerShell)
.\venv\Scripts\activate

# Ativar (Linux/Mac)
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure a chave da API

Crie um arquivo `.env` na raiz do projeto:

```
GOOGLE_API_KEY=sua_chave_aqui
ADMIN_PASSWORD=sua_senha_aqui
```

### 5. Execute o projeto

```bash
streamlit run app.py
```

O chatbot abre em **http://localhost:8501**. O painel admin fica acessível pela navegação lateral.

A base vetorial é construída automaticamente no primeiro acesso.

## Estrutura do Projeto

```
chatbot-ebp/
├── app.py                  # Chatbot (página inicial)
├── config.py               # Configurações e variáveis de ambiente
├── requirements.txt        # Dependências Python
├── packages.txt            # Dependências do sistema (para deploy)
├── .env                    # Chaves de API (não versionado)
├── .gitignore
├── .streamlit/
│   └── config.toml         # Configuração do Streamlit
├── pages/
│   └── 1_Admin.py          # Painel administrativo (protegido por senha)
├── rag/
│   ├── ingest.py           # Ingestão e embedding de documentos
│   ├── retriever.py        # Busca semântica no ChromaDB
│   └── prompt.py           # Templates de prompt para o Gemini
├── llm/
│   └── gemini.py           # Integração com Gemini via LangChain
└── data/
    ├── raw/                # Documentos fonte (PDF e TXT)
    │   ├── coordenacao_geral/
    │   ├── coordenacao_estagio/
    │   ├── coordenacao_tfg/
    │   ├── normas_graduacao/
    │   ├── estatutos_regimentos/
    │   ├── projeto_pedagogico/
    │   └── dicas/          # Dicas informais (tratadas como não-oficiais)
    ├── vectorstore/        # Índice vetorial ChromaDB (gerado automaticamente)
    └── dicas.json          # Avisos da coordenação (gerenciado pelo admin)
```

## Funcionalidades

### Chatbot (app.py)
- Chat com IA baseado nos documentos do curso
- Avisos da coordenação na barra lateral
- Exemplos de perguntas para guiar o aluno

### Painel Administrativo (pages/1_Admin.py)
- Upload de documentos (PDF/TXT) por categoria
- Edição de arquivos TXT direto na interface
- Reconstrução do banco vetorial
- Configuração do link e índice do PPC
- Avisos com validade (dias) e opção de manter como conhecimento do chat

## Deploy (Streamlit Community Cloud)

1. Suba o projeto para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Crie um novo app apontando para `app.py`
4. Adicione os secrets (GOOGLE_API_KEY e ADMIN_PASSWORD) nas configurações avançadas
5. A base vetorial será construída automaticamente no primeiro acesso

## Autores

Luiz Carlos Bertucci Barbosa e Helena Esteves Passos — Engenharia de Bioprocessos, UNIFEI
