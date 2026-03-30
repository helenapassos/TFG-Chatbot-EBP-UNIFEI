"""
Módulo de busca semântica: recupera os chunks mais relevantes
do banco vetorial para uma dada pergunta.
"""

from langchain_chroma import Chroma

import config
from rag.ingest import load_vectorstore


def get_retriever(vectorstore: Chroma | None = None):
    """
    Retorna um retriever LangChain configurado para busca por similaridade.
    Se nenhum vectorstore for fornecido, carrega do disco.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K},
    )


def retrieve_context(query: str, vectorstore: Chroma | None = None) -> str:
    """
    Dado uma pergunta, recupera os k chunks mais relevantes e
    retorna o contexto concatenado como string.
    """
    retriever = get_retriever(vectorstore)
    docs = retriever.invoke(query)

    if not docs:
        return ""

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconhecido")
        tipo = doc.metadata.get("tipo", "")
        if tipo == "dica_informal":
            header = f"[Trecho {i} — ⚠️ DICA INFORMAL (NÃO É VERDADE ABSOLUTA) — Fonte: {source}]"
        else:
            header = f"[Trecho {i} — Fonte: {source}]"
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(context_parts)
