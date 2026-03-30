"""
Interface de chat Streamlit para o Chatbot EBP.
"""

import streamlit as st

import config
from config import load_tips_active
from rag.ingest import load_vectorstore, ingest_all
from llm.gemini import ask


def check_vectorstore_exists() -> bool:
    chroma_dir = config.VECTORSTORE_DIR
    return chroma_dir.exists() and any(chroma_dir.iterdir())


def auto_build_vectorstore():
    raw = config.DATA_RAW_DIR
    docs_exist = any(raw.rglob("*.pdf")) or any(raw.rglob("*.txt"))
    if not docs_exist:
        return False

    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    with st.spinner("⏳ Construindo a base de conhecimento pela primeira vez... Isso pode levar alguns minutos."):
        try:
            vs = ingest_all()
            st.session_state.vectorstore = vs
            return True
        except Exception as e:
            st.error(f"Erro ao construir a base: {e}")
            return False


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None


def load_vs():
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = load_vectorstore()
    return st.session_state.vectorstore


logo_url = "https://portalpadrao.ufma.br/ineof/imagens/logo-unifei-oficial.png"

st.markdown(
    f"""
    <style>
    .main-header {{
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }}
    .main-header img {{
        height: 80px;
        margin-bottom: 0.5rem;
    }}
    .main-header h1 {{
        color: #1B5E20;
        font-size: 1.8rem;
        margin-top: 0.3rem;
    }}
    .main-header p {{
        color: #666;
        font-size: 0.95rem;
    }}
    .stChatMessage {{
        border-radius: 12px;
    }}
    </style>
    <div class="main-header">
        <img src="{logo_url}" alt="UNIFEI">
        <h1>Chatbot EBP — UNIFEI</h1>
        <p>Assistente virtual do curso de Engenharia de Bioprocessos</p>
    </div>
    """,
    unsafe_allow_html=True,
)

init_session_state()

if not config.GOOGLE_API_KEY:
    st.error(
        "⚠️ Chave da API do Google não configurada. "
        "Crie um arquivo `.env` com `GOOGLE_API_KEY=sua_chave_aqui`."
    )
    st.stop()

if not check_vectorstore_exists():
    built = auto_build_vectorstore()
    if not built:
        st.warning(
            "📚 A base de conhecimento ainda não foi criada. "
            "Acesse o **Painel Administrativo** na barra lateral para fazer "
            "o upload dos documentos e gerar a base vetorial."
        )
        st.stop()
    st.rerun()

vs = load_vs()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua dúvida sobre o curso de EBP..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando informações..."):
            try:
                response = ask(prompt, vectorstore=vs)
            except Exception as e:
                response = (
                    f"Desculpe, ocorreu um erro ao processar sua pergunta. "
                    f"Tente novamente em alguns instantes.\n\n"
                    f"Detalhes: {e}"
                )
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

with st.sidebar:
    active_tips = load_tips_active()
    if active_tips:
        st.markdown("### 📢 Avisos da Coordenação")
        for tip in active_tips:
            st.info(tip)
        st.divider()

    st.markdown("### Sobre")
    st.markdown(
        "Este chatbot utiliza **inteligência artificial** para responder "
        "dúvidas sobre o curso de **Engenharia de Bioprocessos** da UNIFEI."
    )
    st.markdown(
        "As respostas são baseadas em documentos oficiais do curso. "
        "Para informações não cobertas, procure a coordenação."
    )
    st.divider()
    st.markdown("### Exemplos de perguntas")
    examples = [
        "Quais são os pré-requisitos de TCC1?",
        "Como funciona o estágio obrigatório?",
        "O que conta como atividade complementar?",
        "Qual a carga horária mínima para estágio?",
        "Como faço para trancar uma disciplina?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.messages.append(
                {"role": "user", "content": ex}
            )
            st.rerun()

    st.divider()
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Desenvolvido por Luiz Carlos Bertucci Barbosa e Helena Esteves Passos — UNIFEI")
