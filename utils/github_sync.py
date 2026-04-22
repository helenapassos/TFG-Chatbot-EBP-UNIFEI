"""
GitHub API sync — persiste documentos e vectorstore no repositório git.

Configurar nos Streamlit Secrets (Settings → Secrets):
    GITHUB_TOKEN   = "ghp_..."           # Personal Access Token (scope: repo)
    GITHUB_REPO    = "usuario/repo"      # ex: "davi-vilela/chatbot-ebp"
    GITHUB_BRANCH  = "main"              # branch de deploy
    GITHUB_APP_DIR = "chatbot-ebp"       # subpasta do app no repo (vazio se for a raiz)
"""

import base64
import os
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def _cfg() -> tuple[str, str, str, str]:
    """Retorna (token, repo, branch, app_dir) a partir de env vars ou Streamlit Secrets."""
    token = repo = app_dir = ""
    branch = "main"
    try:
        import streamlit as st
        token    = st.secrets.get("GITHUB_TOKEN", "")
        repo     = st.secrets.get("GITHUB_REPO", "")
        branch   = st.secrets.get("GITHUB_BRANCH", "main") or "main"
        app_dir  = st.secrets.get("GITHUB_APP_DIR", "chatbot-ebp")
    except Exception:
        pass
    token   = os.getenv("GITHUB_TOKEN", token)
    repo    = os.getenv("GITHUB_REPO", repo)
    branch  = os.getenv("GITHUB_BRANCH", branch) or "main"
    app_dir = os.getenv("GITHUB_APP_DIR", app_dir)
    return token, repo, branch, app_dir


def github_configured() -> bool:
    """Retorna True se GITHUB_TOKEN e GITHUB_REPO estiverem configurados."""
    token, repo, _, _ = _cfg()
    return bool(token and repo)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _auth_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Content-Type": "application/json"}


def _get_sha(token: str, repo: str, path: str, branch: str) -> str | None:
    """Retorna o SHA do arquivo no repo, ou None se não existir."""
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers={"Authorization": f"token {token}"},
        params={"ref": branch},
        timeout=15,
    )
    return resp.json().get("sha") if resp.status_code == 200 else None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def commit_file(local_path: Path, repo_path: str, message: str) -> bool:
    """
    Cria ou atualiza um arquivo no repositório GitHub.
    repo_path: caminho relativo no repo (ex: "chatbot-ebp/data/raw/dicas/aviso.txt")
    """
    token, repo, branch, _ = _cfg()
    if not token or not repo:
        return False

    content_b64 = base64.b64encode(local_path.read_bytes()).decode()
    sha = _get_sha(token, repo, repo_path, branch)

    payload: dict = {"message": message, "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha

    resp = requests.put(
        f"https://api.github.com/repos/{repo}/contents/{repo_path}",
        headers=_auth_headers(token),
        json=payload,
        timeout=30,
    )
    return resp.status_code in (200, 201)


def delete_file(repo_path: str, message: str) -> bool:
    """Remove um arquivo do repositório GitHub."""
    token, repo, branch, _ = _cfg()
    if not token or not repo:
        return False

    sha = _get_sha(token, repo, repo_path, branch)
    if sha is None:
        return True  # arquivo já não existe no repo

    resp = requests.delete(
        f"https://api.github.com/repos/{repo}/contents/{repo_path}",
        headers=_auth_headers(token),
        json={"message": message, "sha": sha, "branch": branch},
        timeout=15,
    )
    return resp.status_code == 200


def _list_repo_files(token: str, repo: str, repo_dir: str, branch: str) -> list:
    """Retorna lista de caminhos de todos os arquivos dentro de repo_dir no GitHub."""
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/git/trees/{branch}",
        headers={"Authorization": f"token {token}"},
        params={"recursive": "1"},
        timeout=20,
    )
    if resp.status_code != 200:
        return []
    prefix = repo_dir.rstrip("/") + "/"
    return [
        item["path"]
        for item in resp.json().get("tree", [])
        if item["type"] == "blob" and item["path"].startswith(prefix)
    ]


def commit_directory(
    local_dir: Path,
    repo_dir: str,
    message: str,
    delete_removed: bool = False,
) -> tuple[int, int]:
    """
    Commita todos os arquivos de um diretório em um único commit via Git Data API.
    Se delete_removed=True, remove do repo arquivos que não existem mais localmente.
    Retorna (arquivos_incluídos, falhas).
    """
    token, repo, branch, _ = _cfg()
    if not token or not repo:
        return 0, 0

    headers = _auth_headers(token)

    # 1. SHA do commit atual do branch
    ref_resp = requests.get(
        f"https://api.github.com/repos/{repo}/git/refs/heads/{branch}",
        headers=headers, timeout=15,
    )
    if ref_resp.status_code != 200:
        return 0, 1
    base_sha = ref_resp.json()["object"]["sha"]

    # 2. SHA da tree base
    commit_resp = requests.get(
        f"https://api.github.com/repos/{repo}/git/commits/{base_sha}",
        headers=headers, timeout=15,
    )
    if commit_resp.status_code != 200:
        return 0, 1
    base_tree_sha = commit_resp.json()["tree"]["sha"]

    # 3. Cria blobs para cada arquivo local e monta a tree
    tree_items = []
    local_rel_paths = set()
    fail = 0

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir).as_posix()
        local_rel_paths.add(rel)
        content_b64 = base64.b64encode(file_path.read_bytes()).decode()
        blob_resp = requests.post(
            f"https://api.github.com/repos/{repo}/git/blobs",
            headers=headers,
            json={"content": content_b64, "encoding": "base64"},
            timeout=30,
        )
        if blob_resp.status_code != 201:
            fail += 1
            continue
        tree_items.append({
            "path": f"{repo_dir}/{rel}",
            "mode": "100644",
            "type": "blob",
            "sha": blob_resp.json()["sha"],
        })

    # 4. Arquivos removidos localmente → deletar da tree
    if delete_removed:
        remote_paths = _list_repo_files(token, repo, repo_dir, branch)
        prefix = repo_dir.rstrip("/") + "/"
        for remote_path in remote_paths:
            rel = remote_path[len(prefix):]
            if rel not in local_rel_paths:
                tree_items.append({
                    "path": remote_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": None,  # sha=None deleta o arquivo na tree
                })

    if not tree_items:
        return 0, fail

    # 5. Cria nova tree
    tree_resp = requests.post(
        f"https://api.github.com/repos/{repo}/git/trees",
        headers=headers,
        json={"base_tree": base_tree_sha, "tree": tree_items},
        timeout=30,
    )
    if tree_resp.status_code != 201:
        return 0, fail + 1
    new_tree_sha = tree_resp.json()["sha"]

    # 6. Cria o commit
    new_commit_resp = requests.post(
        f"https://api.github.com/repos/{repo}/git/commits",
        headers=headers,
        json={"message": message, "tree": new_tree_sha, "parents": [base_sha]},
        timeout=15,
    )
    if new_commit_resp.status_code != 201:
        return 0, fail + 1
    new_commit_sha = new_commit_resp.json()["sha"]

    # 7. Atualiza o ref do branch
    update_resp = requests.patch(
        f"https://api.github.com/repos/{repo}/git/refs/heads/{branch}",
        headers=headers,
        json={"sha": new_commit_sha},
        timeout=15,
    )
    if update_resp.status_code != 200:
        return 0, fail + 1

    return len(tree_items), fail


# ---------------------------------------------------------------------------
# Helpers de caminho para este projeto
# ---------------------------------------------------------------------------

def _prefix() -> str:
    _, _, _, app_dir = _cfg()
    return f"{app_dir}/" if app_dir else ""


def raw_doc_repo_path(category: str, filename: str) -> str:
    """Caminho no repo para um documento da base de conhecimento."""
    return f"{_prefix()}data/raw/{category}/{filename}"


def vectorstore_repo_dir() -> str:
    """Prefixo do diretório do vectorstore no repo."""
    return f"{_prefix()}data/vectorstore"


def ppc_config_repo_path() -> str:
    """Caminho no repo para o arquivo de configuração do PPC."""
    return f"{_prefix()}data/ppc_config.json"


def tips_repo_path() -> str:
    """Caminho no repo para o arquivo de dicas/avisos."""
    return f"{_prefix()}data/dicas.json"
