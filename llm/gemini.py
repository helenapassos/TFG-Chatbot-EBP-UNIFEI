"""
Wrapper da API do Google Gemini via LangChain.
Gerencia a criação da chain RAG e a geração de respostas,
com rotação automática de chaves de API ao atingir cota.
"""

import logging
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config
from rag.prompt import get_chat_prompt
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)

# Índice da chave em uso — rotaciona automaticamente ao esgotar cota
_current_key_index = 0


def _is_quota_error(exc: Exception) -> bool:
    """Verifica se a exceção indica cota/rate limit esgotado."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("quota", "resource_exhausted", "resourceexhausted", "429", "rate limit")
    )


def _is_transient_error(exc: Exception) -> bool:
    """Verifica se é erro temporário do servidor (503, sobrecarga)."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("503", "service unavailable", "unavailable", "overloaded", "high demand", "try again")
    )


def get_llm(api_key: str = "") -> ChatGoogleGenerativeAI:
    """Retorna a instância do modelo Gemini configurada."""
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=api_key or config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        convert_system_message_to_human=False,
    )


def get_rag_chain(vectorstore=None, api_key: str = ""):
    """
    Monta a chain RAG completa:
    pergunta → retriever → prompt + contexto + histórico → LLM → resposta.
    """
    retriever = get_retriever(vectorstore)
    prompt = get_chat_prompt()
    llm = get_llm(api_key)

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
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str, vectorstore=None, history: list = None) -> str:
    """
    Envia uma pergunta ao chatbot e retorna a resposta.
    history: lista de HumanMessage/AIMessage com os últimos turnos da conversa.
    Rotaciona automaticamente entre as chaves de API configuradas
    quando uma chave esgota sua cota.
    """
    global _current_key_index

    keys = config.GOOGLE_API_KEYS
    if not keys:
        raise ValueError(
            "Nenhuma GOOGLE_API_KEY configurada. "
            "Defina GOOGLE_API_KEY ou GOOGLE_API_KEYS no .env ou nos secrets."
        )

    payload = {"question": question, "history": history or []}
    last_exc = None

    for attempt in range(len(keys)):
        current_key = keys[_current_key_index]
        try:
            chain = get_rag_chain(vectorstore, api_key=current_key)
            response = chain.invoke(payload)
            time.sleep(1)
            return response
        except Exception as exc:
            if _is_transient_error(exc):
                raise RuntimeError(
                    "O serviço de IA está temporariamente sobrecarregado. "
                    "Tente reformular sua pergunta com mais detalhes sobre o que você precisa saber."
                ) from exc
            elif _is_quota_error(exc):
                next_index = (_current_key_index + 1) % len(keys)
                logger.warning(
                    "Cota esgotada na chave %d/%d. Alternando para chave %d...",
                    _current_key_index + 1,
                    len(keys),
                    next_index + 1,
                )
                _current_key_index = next_index
                last_exc = exc
            else:
                raise

    raise RuntimeError(
        f"Todas as {len(keys)} chave(s) de API estão com cota esgotada. "
        "Aguarde a renovação ou adicione mais chaves em GOOGLE_API_KEYS."
    ) from last_exc
