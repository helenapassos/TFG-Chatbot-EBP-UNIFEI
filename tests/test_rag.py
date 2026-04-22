"""
Testes do pipeline RAG do Chatbot EBP.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.prompt import get_chat_prompt, SYSTEM_TEMPLATE
from rag.ingest import _split_documents


class TestPrompt:
    """Testes para o módulo de engenharia de prompt."""

    def test_prompt_template_has_context_variable(self):
        prompt = get_chat_prompt()
        input_vars = prompt.input_variables
        assert "context" in input_vars

    def test_prompt_template_has_question_variable(self):
        prompt = get_chat_prompt()
        input_vars = prompt.input_variables
        assert "question" in input_vars

    def test_system_template_restricts_scope(self):
        assert "APENAS" in SYSTEM_TEMPLATE
        assert "não possui essa informação" in SYSTEM_TEMPLATE.lower() or \
               "Não invente" in SYSTEM_TEMPLATE

    def test_system_template_mentions_ebp(self):
        assert "Engenharia de Bioprocessos" in SYSTEM_TEMPLATE
        assert "UNIFEI" in SYSTEM_TEMPLATE


class TestChunking:
    """Testes para a divisão de documentos em chunks."""

    def test_split_creates_chunks(self):
        from langchain_core.documents import Document
        docs = [Document(page_content="A " * 500, metadata={"source": "test.txt"})]
        chunks = _split_documents(docs)
        assert len(chunks) >= 1

    def test_chunks_have_metadata(self):
        from langchain_core.documents import Document
        docs = [Document(page_content="Texto " * 200, metadata={"source": "doc.pdf"})]
        chunks = _split_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata


class TestConfig:
    """Testes para as configurações do projeto."""

    def test_data_dir_path(self):
        assert config.DATA_RAW_DIR.name == "raw"
        assert config.DATA_RAW_DIR.parent.name == "data"

    def test_vectorstore_dir_path(self):
        assert config.VECTORSTORE_DIR.name == "vectorstore"

    def test_raw_subdirs_not_empty(self):
        assert len(config.RAW_SUBDIRS) > 0

    def test_chunk_size_reasonable(self):
        assert 200 <= config.CHUNK_SIZE <= 2000

    def test_retriever_k_reasonable(self):
        assert 1 <= config.RETRIEVER_K <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
