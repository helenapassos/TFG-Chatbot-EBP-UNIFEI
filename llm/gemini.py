"""
Wrapper da API do Google Gemini via LangChain.
Gerencia a criação da chain RAG e a geração de respostas.
"""

import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config
from rag.prompt import get_chat_prompt
from rag.retriever import get_retriever


def get_llm() -> ChatGoogleGenerativeAI:
    """Retorna a instância do modelo Gemini configurada."""
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        convert_system_message_to_human=False,
    )


def get_rag_chain(vectorstore=None):
    """
    Monta a chain RAG completa:
    pergunta → retriever → prompt + contexto → LLM → resposta.
    """
    retriever = get_retriever(vectorstore)
    prompt = get_chat_prompt()
    llm = get_llm()

    def format_docs(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconhecido")
            tipo = doc.metadata.get("tipo", "")
            if tipo == "dica_informal":
                header = (
                    f"[Trecho {i} — ⚠️ DICA INFORMAL (NÃO É VERDADE ABSOLUTA) — Fonte: {source}]"
                )
            else:
                header = f"[Trecho {i} — Fonte: {source}]"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str, vectorstore=None) -> str:
    """
    Envia uma pergunta ao chatbot e retorna a resposta.
    Inclui rate limiting para respeitar o tier gratuito da API.
    """
    chain = get_rag_chain(vectorstore)
    response = chain.invoke(question)
    time.sleep(1)
    return response
