"""
Ingestão de documentos: carrega PDFs e TXTs, divide em chunks e
armazena os embeddings no banco vetorial ChromaDB.

Suporta ingestão incremental: apenas arquivos novos ou alterados são
reembedados, economizando tokens da API.
"""

import hashlib
import json
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import logging

import config

logger = logging.getLogger(__name__)

BATCH_SIZE = 80
BATCH_DELAY = 62

# Índice da chave de embedding em uso — rotaciona ao esgotar cota
_current_key_index = 0


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("quota", "resource_exhausted", "resourceexhausted", "429", "rate limit"))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(api_key: str = "") -> GoogleGenerativeAIEmbeddings:
    """Retorna o modelo de embeddings configurado."""
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=api_key or config.GOOGLE_API_KEY,
    )


def _make_vectorstore(api_key: str = "") -> Chroma:
    """Cria instância do Chroma com a chave fornecida."""
    return Chroma(
        embedding_function=get_embeddings(api_key),
        persist_directory=str(config.VECTORSTORE_DIR),
        collection_name="ebp_knowledge_base",
    )


def reset_vectorstore() -> None:
    """Apaga a coleção do ChromaDB via API (sem deletar arquivos).

    Evita o PermissionError do Windows ao tentar deletar chroma.sqlite3
    enquanto outra conexão ainda o mantém aberto.
    Após este reset, ingest_all() cria a coleção do zero.
    """
    try:
        vs = _make_vectorstore()
        vs.delete_collection()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers de carregamento e divisão
# ---------------------------------------------------------------------------

def _load_single_document(file_path: Path) -> list:
    """Carrega documentos de um único arquivo PDF ou TXT."""
    if file_path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    elif file_path.suffix.lower() == ".txt":
        return TextLoader(str(file_path), encoding="utf-8").load()
    return []


def _load_documents(directory: Path) -> list:
    """Carrega todos os PDFs e TXTs de um diretório (recursivo)."""
    docs = []
    for file_path in sorted(directory.rglob("*")):
        docs.extend(_load_single_document(file_path))
    return docs


def _split_documents(docs: list) -> list:
    """Divide documentos em chunks menores."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _tag_dicas(chunks: list) -> list:
    """Marca chunks originários da pasta 'dicas' com metadata tipo=dica_informal."""
    dicas_path = str(config.DATA_RAW_DIR / config.DICAS_SUBDIR).replace("\\", "/").lower()
    for chunk in chunks:
        source = chunk.metadata.get("source", "").replace("\\", "/").lower()
        if dicas_path in source:
            chunk.metadata["tipo"] = "dica_informal"
    return chunks


def _add_chunks_in_batches(vectorstore: Chroma, chunks: list, progress_callback=None) -> Chroma:
    """Adiciona chunks ao vectorstore em lotes, respeitando rate limit.

    Rotaciona automaticamente entre as chaves de API ao esgotar cota.
    Retorna o vectorstore (pode ser nova instância se a chave rotacionou).
    progress_callback(msg: str, fraction: float) é chamado a cada lote se fornecido.
    """
    global _current_key_index
    keys = config.GOOGLE_API_KEYS
    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    i = 0
    while i < total:
        batch = chunks[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Lote {batch_num}/{total_batches} ({len(batch)} chunks)...")
        try:
            vectorstore.add_documents(batch)
        except Exception as exc:
            if _is_quota_error(exc) and len(keys) > 1:
                next_index = (_current_key_index + 1) % len(keys)
                logger.warning(
                    "Cota de embedding esgotada na chave %d/%d. Alternando para chave %d...",
                    _current_key_index + 1, len(keys), next_index + 1,
                )
                if progress_callback:
                    progress_callback(
                        f"Cota esgotada — alternando para chave {next_index + 1}/{len(keys)}...",
                        i / total,
                    )
                _current_key_index = next_index
                vectorstore = _make_vectorstore(api_key=keys[_current_key_index])
                continue  # retry o mesmo lote com a nova chave
            raise

        if progress_callback:
            progress_callback(
                f"Lote {batch_num}/{total_batches} — {i + len(batch)}/{total} chunks indexados",
                (i + len(batch)) / total,
            )
        i += BATCH_SIZE
        if i < total:
            print(f"  Aguardando {BATCH_DELAY}s (rate limit)...")
            if progress_callback:
                for remaining in range(BATCH_DELAY, 0, -5):
                    progress_callback(
                        f"Aguardando {remaining}s antes do próximo lote...",
                        i / total,
                    )
                    time.sleep(min(5, remaining))
            else:
                time.sleep(BATCH_DELAY)
    return vectorstore


# ---------------------------------------------------------------------------
# Manifesto — rastreia quais arquivos já foram ingeridos
# ---------------------------------------------------------------------------

def _manifest_path() -> Path:
    return config.VECTORSTORE_DIR / "ingest_manifest.json"


def _file_hash(path: Path) -> str:
    """Hash MD5 do conteúdo do arquivo."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    mp = _manifest_path()
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_manifest(manifest: dict) -> None:
    mp = _manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ingestão incremental (padrão — econômica)
# ---------------------------------------------------------------------------

def ingest_incremental(progress_callback=None) -> tuple:
    """
    Ingestão incremental: processa apenas arquivos novos ou alterados.

    Compara cada arquivo com o manifesto salvo (hash MD5). Arquivos sem
    mudança são ignorados — zero tokens gastos com eles.

    Retorna (vectorstore, stats) onde stats é um dict com:
      - skipped     : arquivos ignorados (sem mudança)
      - added_files : arquivos novos ou alterados reembedados
      - removed_files: arquivos deletados do disco cujos chunks foram removidos
      - added_chunks : total de novos chunks gerados
    """
    vectorstore = _make_vectorstore(api_key=config.GOOGLE_API_KEYS[_current_key_index] if config.GOOGLE_API_KEYS else "")
    manifest = _load_manifest()

    # Mapeia arquivos atuais no disco
    current_files = {}
    for file_path in sorted(config.DATA_RAW_DIR.rglob("*")):
        if file_path.suffix.lower() in (".pdf", ".txt"):
            current_files[str(file_path)] = _file_hash(file_path)

    # Arquivos removidos do disco → deletar chunks do vectorstore
    removed_keys = set(manifest.keys()) - set(current_files.keys())
    for path_str in removed_keys:
        try:
            vectorstore.delete(where={"source": path_str})
            print(f"  Removido do vectorstore: {Path(path_str).name}")
        except Exception:
            pass

    # Arquivos novos ou com hash diferente → reingerir
    to_ingest = [
        path_str for path_str, h in current_files.items()
        if manifest.get(path_str) != h
    ]

    stats = {
        "skipped": len(current_files) - len(to_ingest),
        "added_files": len(to_ingest),
        "removed_files": len(removed_keys),
        "added_chunks": 0,
    }

    if not to_ingest:
        print("Nenhum arquivo novo ou alterado. Vectorstore já está atualizado.")
        if progress_callback:
            progress_callback("Base vetorial já está atualizada.", 1.0)
        _save_manifest({k: current_files[k] for k in current_files})
        return vectorstore, stats

    print(f"{len(to_ingest)} arquivo(s) novo(s)/alterado(s). {stats['skipped']} ignorado(s).")

    all_new_chunks = []
    for idx, path_str in enumerate(to_ingest):
        name = Path(path_str).name
        if progress_callback:
            progress_callback(f"Lendo {name} ({idx + 1}/{len(to_ingest)})...", 0.0)
        if path_str in manifest:
            try:
                vectorstore.delete(where={"source": path_str})
            except Exception:
                pass
        docs = _load_single_document(Path(path_str))
        chunks = _split_documents(docs)
        all_new_chunks.extend(chunks)
        print(f"  {name}: {len(chunks)} chunks")

    all_new_chunks = _tag_dicas(all_new_chunks)
    stats["added_chunks"] = len(all_new_chunks)

    print(f"Total de chunks novos: {len(all_new_chunks)}. Processando em lotes de {BATCH_SIZE}...")
    if progress_callback:
        progress_callback(f"Iniciando embedding de {len(all_new_chunks)} chunks...", 0.0)
    vectorstore = _add_chunks_in_batches(vectorstore, all_new_chunks, progress_callback=progress_callback)

    _save_manifest({k: current_files[k] for k in current_files})
    return vectorstore, stats


# ---------------------------------------------------------------------------
# Ingestão completa (força reembedar tudo — usar só quando necessário)
# ---------------------------------------------------------------------------

def ingest_all(progress_callback=None) -> Chroma:
    """
    Pipeline completo de ingestão: carrega, divide e armazena no ChromaDB.
    Reembeda TODOS os documentos do zero — use apenas quando necessário,
    pois consome mais tokens. Prefira ingest_incremental() no dia a dia.

    progress_callback(msg: str, fraction: float) é chamado a cada etapa se fornecido.
    """
    all_files = sorted(
        p for p in config.DATA_RAW_DIR.rglob("*")
        if p.suffix.lower() in (".pdf", ".txt")
    )
    if not all_files:
        raise FileNotFoundError(
            f"Nenhum documento encontrado em {config.DATA_RAW_DIR}. "
            "Adicione arquivos .pdf ou .txt nas subpastas de data/raw/."
        )

    # Leitura é operação de disco — muito rápida para exibir por arquivo.
    # Mostra uma mensagem única com o total e lista os nomes no console.
    if progress_callback:
        progress_callback(f"Carregando {len(all_files)} arquivo(s)...", 0.02)

    docs = []
    for file_path in all_files:
        print(f"  Lendo {file_path.name}...")
        docs.extend(_load_single_document(file_path))

    if progress_callback:
        progress_callback(f"Dividindo {len(docs)} página(s) em chunks...", 0.08)

    chunks = _split_documents(docs)
    chunks = _tag_dicas(chunks)
    total = len(chunks)
    print(f"Total de chunks: {total}. Processando em lotes de {BATCH_SIZE}...")

    if progress_callback:
        progress_callback(f"Iniciando embedding de {total} chunks em lotes de {BATCH_SIZE}...", 0.12)

    vectorstore = _make_vectorstore(api_key=config.GOOGLE_API_KEYS[_current_key_index] if config.GOOGLE_API_KEYS else "")
    vectorstore = _add_chunks_in_batches(vectorstore, chunks, progress_callback=progress_callback)

    if progress_callback:
        progress_callback("Salvando manifesto...", 1.0)

    # Reconstrói o manifesto com todos os arquivos atuais
    manifest = {}
    for file_path in all_files:
        manifest[str(file_path)] = _file_hash(file_path)
    _save_manifest(manifest)

    return vectorstore


# ---------------------------------------------------------------------------
# Carregamento do vectorstore existente
# ---------------------------------------------------------------------------

def load_vectorstore() -> Chroma:
    """Carrega o banco vetorial existente do disco.
    Usa a mesma chave de embedding que está ativa no momento (respeita rotação).
    """
    embeddings = get_embeddings(
        api_key=config.GOOGLE_API_KEYS[_current_key_index] if config.GOOGLE_API_KEYS else ""
    )
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
