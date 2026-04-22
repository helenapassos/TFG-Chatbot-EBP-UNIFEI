import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _is_writable(path: Path) -> bool:
    """Verifica se um diretório (ou seu pai) permite escrita."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _resolve_dir(repo_path: Path, tmp_name: str) -> Path:
    """Retorna repo_path se gravável, senão /tmp/<tmp_name>."""
    if _is_writable(repo_path):
        return repo_path
    tmp = Path("/tmp") / tmp_name
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def _get_secret(key: str, default: str = "") -> str:
    """Busca segredo do .env ou dos Streamlit Secrets (para deploy na nuvem)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


def _get_api_keys() -> list:
    """Retorna lista de chaves da API do Google.

    Aceita três formas de configuração:
    - GOOGLE_API_KEYS=chave1,chave2,chave3  (string separada por vírgula — .env)
    - GOOGLE_API_KEYS = ["chave1", "chave2"] (lista TOML — Streamlit Secrets)
    - GOOGLE_API_KEY=chave                   (chave única — retrocompatível)
    """
    keys_val = _get_secret("GOOGLE_API_KEYS", "")
    if keys_val:
        if isinstance(keys_val, list):
            # Streamlit Secrets retornou um array TOML
            keys = [k.strip() for k in keys_val if isinstance(k, str) and k.strip()]
        else:
            # String separada por vírgula (do .env ou secrets como string)
            keys = [k.strip() for k in str(keys_val).split(",") if k.strip()]
        if keys:
            return keys
    single = _get_secret("GOOGLE_API_KEY", "")
    return [single] if single else []


GOOGLE_API_KEYS: list = _get_api_keys()
GOOGLE_API_KEY: str = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""  # retrocompatível
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", "admin123")

# Diretórios do repositório (podem ser read-only no Streamlit Cloud)
REPO_DATA_RAW_DIR = BASE_DIR / "data" / "raw"
REPO_VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

# Diretórios efetivos: usam /tmp como fallback se o repo for read-only
DATA_RAW_DIR = _resolve_dir(REPO_DATA_RAW_DIR, "ebp_raw")
VECTORSTORE_DIR = _resolve_dir(REPO_VECTORSTORE_DIR, "ebp_vectorstore")

PPC_CONFIG_PATH = BASE_DIR / "data" / "ppc_config.json"
TIPS_PATH = BASE_DIR / "data" / "dicas.json"

RAW_SUBDIRS = [
    "coordenacao_geral",
    "coordenacao_estagio",
    "coordenacao_tfg",
    "normas_graduacao",
    "estatutos_regimentos",
    "projeto_pedagogico",
    "dicas",
]

DICAS_SUBDIR = "dicas"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3
RETRIEVER_K = 5

_DEFAULT_PPC_LINK = (
    "https://sigaa.unifei.edu.br/sigaa/verProducao"
    "?idProducao=2093997&&key=dcb084ddb86fe97c5ca41630ed63ebb1"
)

_DEFAULT_PPC_SECTIONS = """\
1. Introdução (1.1 A UNIFEI; 1.2 O curso de EBP — visão geral, Biotecnologia, profissão, contexto)
2. Dados do Curso (regime, duração, vagas, carga horária)
3. Objetivos do Curso
4. Formas de Acesso (4.1 Admissão inicial — ENEM/SiSU; 4.2 Admissão complementar — transferências)
5. Competências, Habilidades e Perfil do Egresso (5.1 Intrínsecas; 5.2 DCN's; 5.3 Perfil)
6. Fundamentos Didático-Pedagógicos (6.1 Acolhimento; 6.2 Metodologias de ensino)
7. Sistemas de Avaliação (7.1 Rendimento escolar — notas, frequência, substitutiva; 7.2 Avaliação do PPC)
8. Corpo Docente
9. Órgãos Administrativos (9.1 Coordenação; 9.2 Colegiado; 9.3 NDE)
10. Infraestrutura Física (laboratórios, biblioteca)
11. Estrutura Curricular (11.1 Matriz curricular; 11.2 Competências por disciplina; 11.3 Optativas; \
11.4 Extensão; 11.5 Atividades Complementares; 11.6 Estágio e TCC; 11.7 Trilhas de conhecimento; \
11.8 Ementário — ementa, pré-requisitos e bibliografia de cada disciplina)
12. Referências"""


def load_ppc_config() -> dict:
    """Carrega link e seções do PPC a partir do arquivo JSON, ou retorna defaults."""
    if PPC_CONFIG_PATH.exists():
        try:
            data = json.loads(PPC_CONFIG_PATH.read_text(encoding="utf-8"))
            return {
                "ppc_link": data.get("ppc_link", _DEFAULT_PPC_LINK),
                "ppc_sections": data.get("ppc_sections", _DEFAULT_PPC_SECTIONS),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"ppc_link": _DEFAULT_PPC_LINK, "ppc_sections": _DEFAULT_PPC_SECTIONS}


def save_ppc_config(ppc_link: str, ppc_sections: str) -> None:
    """Salva link e seções do PPC no arquivo JSON."""
    PPC_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PPC_CONFIG_PATH.write_text(
        json.dumps({"ppc_link": ppc_link, "ppc_sections": ppc_sections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_tips_raw() -> list[dict]:
    """Carrega todas as dicas como objetos do arquivo JSON.

    Cada dica é um dict com:
      - text (str): conteúdo da dica
      - created_at (str): data de criação ISO (YYYY-MM-DD)
      - days (int): dias de exibição na sidebar (0 = permanente)
      - keep_as_knowledge (bool): manter como base de conhecimento após expirar
    """
    if TIPS_PATH.exists():
        try:
            data = json.loads(TIPS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                tips = []
                for item in data:
                    if isinstance(item, str) and item.strip():
                        tips.append({
                            "text": item.strip(),
                            "created_at": datetime.date.today().isoformat(),
                            "days": 0,
                            "keep_as_knowledge": True,
                        })
                    elif isinstance(item, dict) and item.get("text", "").strip():
                        tips.append(item)
                return tips
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_tips_raw(tips: list[dict]) -> None:
    """Salva a lista de dicas (objetos) no arquivo JSON."""
    TIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIPS_PATH.write_text(
        json.dumps(tips, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_tip_active(tip: dict) -> bool:
    """Verifica se a dica ainda está dentro do prazo de exibição."""
    days = tip.get("days", 0)
    if days == 0:
        return True
    try:
        created = datetime.date.fromisoformat(tip["created_at"])
        return datetime.date.today() <= created + datetime.timedelta(days=days)
    except (KeyError, ValueError):
        return True


def load_tips_active() -> list[str]:
    """Retorna apenas os textos das dicas ativas (não expiradas) para a sidebar."""
    return [t["text"] for t in _load_tips_raw() if _is_tip_active(t)]


def load_tips_for_prompt() -> list[str]:
    """Retorna textos de dicas para o prompt do LLM: ativas + expiradas com keep_as_knowledge."""
    result = []
    for tip in _load_tips_raw():
        if _is_tip_active(tip) or tip.get("keep_as_knowledge", False):
            result.append(tip["text"])
    return result


def load_tips() -> list[dict]:
    """Carrega todas as dicas como objetos (para o admin)."""
    return _load_tips_raw()


def save_tips(tips: list[dict]) -> None:
    """Salva a lista de dicas (objetos) no arquivo JSON."""
    _save_tips_raw(tips)


def cleanup_expired_tips() -> int:
    """Remove dicas expiradas que NÃO devem ser mantidas como conhecimento. Retorna quantas foram removidas."""
    tips = _load_tips_raw()
    before = len(tips)
    tips = [t for t in tips if _is_tip_active(t) or t.get("keep_as_knowledge", False)]
    _save_tips_raw(tips)
    return before - len(tips)
