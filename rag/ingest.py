"""
Ingestão de documentos: carrega PDFs e TXTs, divide em chunks e
armazena os embeddings no banco vetorial ChromaDB.
"""

import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import config

BATCH_SIZE = 80
BATCH_DELAY = 62


def _load_documents(directory: Path) -> list:
    """Carrega todos os PDFs e TXTs de um diretório (recursivo)."""
    docs = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def _split_documents(docs: list) -> list:
    """Divide documentos em chunks menores."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Retorna o modelo de embeddings configurado."""
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )


def _tag_dicas(chunks: list) -> list:
    """Marca chunks originários da pasta 'dicas' com metadata tipo=dica_informal."""
    dicas_path = str(config.DATA_RAW_DIR / config.DICAS_SUBDIR).replace("\\", "/").lower()
    for chunk in chunks:
        source = chunk.metadata.get("source", "").replace("\\", "/").lower()
        if dicas_path in source:
            chunk.metadata["tipo"] = "dica_informal"
    return chunks


def ingest_all() -> Chroma:
    """Pipeline completo de ingestão: carrega, divide e armazena no ChromaDB."""
    docs = _load_documents(config.DATA_RAW_DIR)
    if not docs:
        raise FileNotFoundError(
            f"Nenhum documento encontrado em {config.DATA_RAW_DIR}. "
            "Adicione arquivos .pdf ou .txt nas subpastas de data/raw/."
        )

    chunks = _split_documents(docs)
    chunks = _tag_dicas(chunks)
    embeddings = get_embeddings()
    total = len(chunks)
    print(f"Total de chunks: {total}. Processando em lotes de {BATCH_SIZE}...")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
        collection_name="ebp_knowledge_base",
    )

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Lote {batch_num}/{total_batches} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)
        if i + BATCH_SIZE < total:
            print(f"  Aguardando {BATCH_DELAY}s (rate limit)...")
            time.sleep(BATCH_DELAY)

    return vectorstore


def load_vectorstore() -> Chroma:
    """Carrega o banco vetorial existente do disco."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(config.VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name="ebp_knowledge_base",
    )


if __name__ == "__main__":
    print("Iniciando ingestão de documentos...")
    vs = ingest_all()
    count = vs._collection.count()
    print(f"Ingestão concluída! {count} chunks armazenados no banco vetorial.")
