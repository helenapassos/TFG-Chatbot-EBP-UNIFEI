"""
Painel Administrativo — Upload e gerenciamento da base de conhecimento.
Acesso protegido por senha para coordenadores.
"""

import datetime

import streamlit as st

import config
from config import (
    load_ppc_config, save_ppc_config, load_tips, save_tips,
    cleanup_expired_tips, _is_tip_active,
)
from rag.ingest import ingest_all, ingest_incremental, load_vectorstore, reset_vectorstore, _load_manifest, _file_hash
from utils.github_sync import (
    github_configured, commit_file, delete_file,
    commit_directory, raw_doc_repo_path, vectorstore_repo_dir,
    ppc_config_repo_path, tips_repo_path,
)


LOGO_URL = "https://portalpadrao.ufma.br/ineof/imagens/logo-unifei-oficial.png"


def _has_pending_changes() -> bool:
    """Retorna True se algum arquivo foi modificado desde o último update da base vetorial."""
    manifest = _load_manifest()
    for file_path in config.DATA_RAW_DIR.rglob("*"):
        if file_path.suffix.lower() not in (".pdf", ".txt"):
            continue
        key = str(file_path)
        if manifest.get(key) != _file_hash(file_path):
            return True
    return False


def check_auth() -> bool:
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    return st.session_state.admin_authenticated


def login_form():
    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <img src="{LOGO_URL}" alt="UNIFEI" style="height: 80px; margin-bottom: 0.5rem;">
            <h1 style="color: #1B5E20; margin-top: 0.3rem;">🔐 Painel Administrativo</h1>
            <p style="color: #666;">Chatbot EBP — UNIFEI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        password = st.text_input("Senha de acesso", type="password")
        submitted = st.form_submit_button("Entrar", use_container_width=True)

        if submitted:
            if password == config.ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Senha incorreta.")


def list_documents():
    st.subheader("📄 Documentos na Base de Conhecimento")

    total_files = 0
    for subdir_name in config.RAW_SUBDIRS:
        subdir = config.DATA_RAW_DIR / subdir_name
        files = list(subdir.glob("*.pdf")) + list(subdir.glob("*.txt"))
        if files:
            with st.expander(f"📁 {subdir_name} ({len(files)} arquivo(s))"):
                for f in sorted(files):
                    col1, col2 = st.columns([4, 1])
                    col1.write(f"📎 {f.name}")
                    if col2.button("🗑️", key=f"del_{f}", help=f"Remover {f.name}"):
                        f.unlink()
                        if github_configured():
                            with st.spinner(f"Removendo {f.name} do repositório..."):
                                delete_file(
                                    raw_doc_repo_path(subdir_name, f.name),
                                    f"chore: remove {f.name} via admin panel",
                                )
                        st.success(f"{f.name} removido.")
                        st.rerun()
            total_files += len(files)

    if total_files == 0:
        st.info("Nenhum documento encontrado. Faça o upload de arquivos abaixo.")

    return total_files


def upload_section():
    st.subheader("📤 Upload de Documentos")

    category = st.selectbox(
        "Categoria do documento",
        options=config.RAW_SUBDIRS,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    uploaded_files = st.file_uploader(
        "Selecione os arquivos (PDF ou TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("💾 Salvar documentos", use_container_width=True):
        dest_dir = config.DATA_RAW_DIR / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        gh = github_configured()
        for uploaded_file in uploaded_files:
            dest_path = dest_dir / uploaded_file.name
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if gh:
                with st.spinner(f"Salvando {uploaded_file.name} no repositório..."):
                    ok = commit_file(
                        dest_path,
                        raw_doc_repo_path(category, uploaded_file.name),
                        f"chore: add {uploaded_file.name} via admin panel",
                    )
                if ok:
                    st.success(f"✅ {uploaded_file.name} salvo e commitado no repositório.")
                else:
                    st.warning(
                        f"✅ {uploaded_file.name} salvo localmente, mas **não foi possível "
                        f"commitar no GitHub**. Verifique o GITHUB_TOKEN nos Secrets."
                    )
            else:
                st.success(f"✅ {uploaded_file.name} salvo em {category}/")

        if not gh:
            st.info(
                "⚠️ GitHub não configurado — arquivos salvos apenas temporariamente. "
                "Configure GITHUB_TOKEN e GITHUB_REPO nos Secrets para persistência permanente."
            )

        st.rerun()


def _sync_vectorstore_to_github(delete_removed: bool = False):
    if github_configured():
        with st.spinner("Salvando vectorstore no repositório (persistência permanente)..."):
            ok, fail = commit_directory(
                config.VECTORSTORE_DIR,
                vectorstore_repo_dir(),
                "chore: update vectorstore via admin panel",
                delete_removed=delete_removed,
            )
        if fail == 0:
            st.success(f"✅ Vectorstore sincronizado no repositório ({ok} arquivo(s)).")
        else:
            st.warning(
                f"⚠️ Vectorstore parcialmente sincronizado: "
                f"{ok} ok, {fail} falha(s). Verifique o GITHUB_TOKEN."
            )
    else:
        st.info(
            "⚠️ GitHub não configurado — vectorstore salvo apenas temporariamente. "
            "Configure GITHUB_TOKEN e GITHUB_REPO nos Secrets para persistência permanente."
        )


def _commit_file_to_github(local_path, repo_path: str, label: str = ""):
    """Commita um arquivo individual para o GitHub. Silencioso se não configurado."""
    if not github_configured():
        return
    name = label or local_path.name
    with st.spinner(f"Salvando {name} no repositório..."):
        ok = commit_file(local_path, repo_path, f"chore: update {name} via admin panel")
    if not ok:
        st.warning(f"Arquivo salvo localmente, mas não foi possível commitar `{name}` no GitHub.")


def rebuild_section():
    st.subheader("🔄 Atualizar Base Vetorial")
    st.markdown(
        "Após adicionar ou remover documentos, atualize o banco vetorial "
        "para que o chatbot utilize as informações atualizadas."
    )

    col_metric, col_btn = st.columns([1, 2])

    with col_metric:
        vs_dir = config.VECTORSTORE_DIR
        if vs_dir.exists() and any(vs_dir.iterdir()):
            try:
                vs = load_vectorstore()
                count = vs._collection.count()
                st.metric("Chunks indexados", count)
            except Exception:
                st.metric("Chunks indexados", "—")
        else:
            st.metric("Chunks indexados", 0)

    with col_btn:
        if st.button(
            "⚡ Atualização incremental",
            use_container_width=True,
            type="primary",
            help="Processa apenas arquivos novos ou alterados. Muito mais rápido e econômico.",
        ):
            progress_bar = st.progress(0.0, text="Iniciando...")
            step_placeholder = st.empty()

            def on_progress(msg: str, fraction: float):
                progress_bar.progress(min(fraction, 1.0), text=msg)
                step_placeholder.caption(f"⏳ {msg}")

            try:
                vs, stats = ingest_incremental(progress_callback=on_progress)
                count = vs._collection.count()
                progress_bar.progress(1.0, text="Concluído!")
                step_placeholder.empty()

                st.session_state.vectorstore = None

                if stats["added_files"] == 0 and stats["removed_files"] == 0:
                    st.success(
                        f"✅ Base já atualizada! "
                        f"{stats['skipped']} arquivo(s) sem mudança — nenhum token gasto."
                    )
                else:
                    parts = []
                    if stats["added_files"]:
                        parts.append(f"{stats['added_files']} arquivo(s) processado(s) ({stats['added_chunks']} chunks)")
                    if stats["removed_files"]:
                        parts.append(f"{stats['removed_files']} arquivo(s) removido(s)")
                    if stats["skipped"]:
                        parts.append(f"{stats['skipped']} ignorado(s) sem mudança")
                    st.success(f"✅ {' • '.join(parts)}. Total: {count} chunks.")
                    _sync_vectorstore_to_github()
            except FileNotFoundError as e:
                progress_bar.empty()
                st.error(str(e))
            except Exception as e:
                progress_bar.empty()
                st.error(f"Erro ao atualizar: {e}")

    st.divider()
    with st.expander("⚠️ Reconstrução completa (reembeda tudo do zero)"):
        st.warning(
            "Use apenas se o banco vetorial estiver corrompido ou se quiser "
            "forçar a reingestão de todos os documentos. "
            "**Consome muito mais tokens da API.**"
        )
        if st.button("🔨 Reconstruir banco vetorial completo", use_container_width=True):
            # Apaga a coleção via API do ChromaDB (sem deletar arquivos —
            # evita PermissionError do Windows com chroma.sqlite3 ainda aberto)
            st.session_state.vectorstore = None
            reset_vectorstore()

            progress_bar = st.progress(0.0, text="Iniciando reconstrução...")
            step_placeholder = st.empty()

            def on_progress_full(msg: str, fraction: float):
                progress_bar.progress(min(fraction, 1.0), text=msg)
                step_placeholder.caption(f"⏳ {msg}")

            try:
                vs = ingest_all(progress_callback=on_progress_full)
                count = vs._collection.count()
                progress_bar.progress(1.0, text="Concluído!")
                step_placeholder.empty()
                st.session_state.vectorstore = None
                st.success(f"✅ Banco vetorial reconstruído! {count} chunks indexados.")
                _sync_vectorstore_to_github(delete_removed=True)
            except FileNotFoundError as e:
                progress_bar.empty()
                st.error(str(e))
            except Exception as e:
                progress_bar.empty()
                st.error(f"Erro ao reconstruir: {e}")


def ppc_settings_section():
    st.subheader("🔗 Link do PPC")
    st.markdown(
        "Quando o chatbot não encontrar uma resposta, ele exibe este link e "
        "sugere a seção do PPC que pode conter a informação."
    )

    ppc = load_ppc_config()

    new_link = st.text_input(
        "URL do Projeto Pedagógico do Curso",
        value=ppc["ppc_link"],
        placeholder="https://...",
    )

    st.divider()
    st.subheader("📑 Índice de Seções do PPC")
    st.markdown(
        "Descreva as seções do documento. O chatbot usa este índice para "
        "indicar ao aluno onde procurar a resposta."
    )

    uploaded_txt = st.file_uploader(
        "Importar índice de seções a partir de um arquivo TXT",
        type=["txt"],
        key="ppc_sections_upload",
    )

    current_sections = ppc["ppc_sections"]
    if uploaded_txt is not None:
        current_sections = uploaded_txt.read().decode("utf-8")
        st.success(f"✅ Conteúdo de '{uploaded_txt.name}' carregado no editor abaixo.")

    new_sections = st.text_area(
        "Índice de seções (edite livremente ou importe pelo upload acima)",
        value=current_sections,
        height=300,
    )

    if st.button("💾 Salvar configurações do PPC", use_container_width=True, type="primary"):
        if not new_link.strip():
            st.error("O link do PPC não pode ficar vazio.")
        elif not new_sections.strip():
            st.error("O índice de seções não pode ficar vazio.")
        else:
            save_ppc_config(new_link.strip(), new_sections.strip())
            _commit_file_to_github(config.PPC_CONFIG_PATH, ppc_config_repo_path(), "ppc_config.json")
            st.success("✅ Configurações do PPC salvas com sucesso! As mudanças já valem para novas perguntas.")


def edit_documents_section():
    st.subheader("✏️ Editar Documentos TXT")
    st.markdown(
        "Selecione um arquivo `.txt` da base de conhecimento para editar "
        "diretamente aqui. Arquivos PDF não podem ser editados por este painel."
    )

    txt_files = []
    for subdir_name in config.RAW_SUBDIRS:
        subdir = config.DATA_RAW_DIR / subdir_name
        for f in sorted(subdir.glob("*.txt")):
            label = f"{subdir_name}/{f.name}"
            txt_files.append((label, f))

    if not txt_files:
        st.info("Nenhum arquivo TXT encontrado na base de conhecimento.")
        return

    labels = [label for label, _ in txt_files]
    selected_label = st.selectbox(
        "Arquivo para editar",
        options=labels,
        format_func=lambda x: f"📄 {x}",
    )

    selected_path = dict(txt_files)[selected_label]

    try:
        current_content = selected_path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return

    st.caption(f"Caminho: `{selected_path.relative_to(config.BASE_DIR)}`")

    edited_content = st.text_area(
        "Conteúdo do arquivo",
        value=current_content,
        height=500,
        key=f"editor_{selected_label}",
    )

    if st.button("💾 Salvar alterações", use_container_width=True, type="primary"):
        if not edited_content.strip():
            st.error("O conteúdo não pode ficar vazio.")
        else:
            selected_path.write_text(edited_content, encoding="utf-8")
            subdir_name = selected_label.split("/")[0]
            _commit_file_to_github(
                selected_path,
                raw_doc_repo_path(subdir_name, selected_path.name),
            )
            st.rerun()

    st.divider()
    st.markdown("##### Criar novo arquivo TXT")

    new_category = st.selectbox(
        "Categoria",
        options=config.RAW_SUBDIRS,
        format_func=lambda x: x.replace("_", " ").title(),
        key="new_txt_category",
    )

    new_filename = st.text_input(
        "Nome do arquivo (sem extensão)",
        placeholder="ex: orientacoes_matricula",
        key="new_txt_filename",
    )

    new_content = st.text_area(
        "Conteúdo inicial",
        height=200,
        placeholder="Digite o conteúdo do novo documento aqui...",
        key="new_txt_content",
    )

    if st.button("📝 Criar arquivo", use_container_width=True):
        if not new_filename.strip():
            st.error("Digite um nome para o arquivo.")
        elif not new_content.strip():
            st.error("O conteúdo não pode ficar vazio.")
        else:
            safe_name = new_filename.strip().replace(" ", "_")
            if not safe_name.endswith(".txt"):
                safe_name += ".txt"
            dest_dir = config.DATA_RAW_DIR / new_category
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / safe_name
            if dest_path.exists():
                st.error(f"Já existe um arquivo com o nome `{safe_name}` nessa categoria.")
            else:
                dest_path.write_text(new_content.strip(), encoding="utf-8")
                _commit_file_to_github(
                    dest_path,
                    raw_doc_repo_path(new_category, safe_name),
                )
                st.success(f"✅ Arquivo `{safe_name}` criado em `{new_category}/`!")
                st.rerun()


def tips_section():
    st.subheader("💡 Avisos e Dicas")
    st.markdown(
        "Adicione avisos e dicas que aparecerão na **barra lateral** do aluno. "
        "Defina por quantos dias o aviso fica visível e se, após expirar, ele "
        "deve ser mantido como base de conhecimento do chatbot."
    )

    tips = load_tips()

    st.divider()
    st.markdown("##### Adicionar novo aviso")

    new_tip = st.text_area(
        "Texto do aviso",
        placeholder="Ex: O período de rematrícula vai de 19/02 a 20/02. Fique atento!",
        height=80,
        key="new_tip_text",
    )

    col_days, col_keep = st.columns(2)
    with col_days:
        new_days = st.number_input(
            "Dias de exibição na barra lateral",
            min_value=0,
            value=7,
            step=1,
            help="0 = permanente (nunca expira)",
            key="new_tip_days",
        )
    with col_keep:
        new_keep = st.checkbox(
            "Manter como conhecimento do chat após expirar",
            value=True,
            help="Se marcado, o chatbot continuará usando essa informação "
                 "para responder perguntas mesmo depois que o aviso sumir da barra lateral.",
            key="new_tip_keep",
        )

    if st.button("➕ Adicionar aviso", use_container_width=True):
        if not new_tip.strip():
            st.error("O texto do aviso não pode ficar vazio.")
        else:
            tips.append({
                "text": new_tip.strip(),
                "created_at": datetime.date.today().isoformat(),
                "days": new_days,
                "keep_as_knowledge": new_keep,
            })
            save_tips(tips)
            _commit_file_to_github(config.TIPS_PATH, tips_repo_path(), "dicas.json")
            st.success("✅ Aviso adicionado!")
            st.rerun()

    st.divider()

    removed = cleanup_expired_tips()
    if removed > 0:
        tips = load_tips()
        st.toast(f"🧹 {removed} aviso(s) expirado(s) removido(s) automaticamente.")

    active = [t for t in tips if _is_tip_active(t)]
    expired_kept = [t for t in tips if not _is_tip_active(t)]

    st.markdown(f"##### 📢 Avisos ativos ({len(active)})")
    if not active:
        st.info("Nenhum aviso ativo no momento.")
    else:
        for i, tip in enumerate(tips):
            if not _is_tip_active(tip):
                continue
            days = tip.get("days", 0)
            created = tip.get("created_at", "—")
            keep = tip.get("keep_as_knowledge", False)

            if days == 0:
                badge = "🟢 Permanente"
            else:
                try:
                    created_date = datetime.date.fromisoformat(created)
                    expires = created_date + datetime.timedelta(days=days)
                    remaining = (expires - datetime.date.today()).days
                    badge = f"⏳ {remaining} dia(s) restante(s)"
                except ValueError:
                    badge = f"⏳ {days} dias"

            keep_label = "✅ Mantém no chat" if keep else "❌ Será deletado"

            with st.container(border=True):
                st.markdown(f"**{tip['text']}**")
                st.caption(f"{badge}  •  {keep_label}  •  Criado em {created}")
                if st.button("🗑️ Remover", key=f"del_tip_{i}", use_container_width=True):
                    tips.pop(i)
                    save_tips(tips)
                    _commit_file_to_github(config.TIPS_PATH, tips_repo_path(), "dicas.json")
                    st.rerun()

    if expired_kept:
        st.divider()
        st.markdown(f"##### 📚 Expirados (mantidos como conhecimento) ({len(expired_kept)})")
        for i, tip in enumerate(tips):
            if _is_tip_active(tip):
                continue
            with st.container(border=True):
                st.markdown(f"**{tip['text']}**")
                st.caption(f"🔇 Expirado  •  Criado em {tip.get('created_at', '—')}")
                if st.button("🗑️ Remover definitivamente", key=f"del_expired_{i}", use_container_width=True):
                    tips.pop(i)
                    save_tips(tips)
                    _commit_file_to_github(config.TIPS_PATH, tips_repo_path(), "dicas.json")
                    st.rerun()


# --- Main ---

if not check_auth():
    login_form()
    st.stop()

st.markdown(
    f"""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <img src="{LOGO_URL}" alt="UNIFEI" style="height: 70px; margin-bottom: 0.5rem;">
        <h1 style="color: #1B5E20; margin-top: 0.3rem;">🔐 Painel Administrativo</h1>
        <p style="color: #666;">Gerenciamento da base de conhecimento — Chatbot EBP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Navegação")
    section = st.radio(
        "Seção",
        [
            "📄 Documentos",
            "📤 Upload",
            "✏️ Editar TXT",
            "🔄 Atualizar Base",
            "⚙️ Config PPC",
            "💡 Dicas",
            "ℹ️ Sobre",
        ],
        label_visibility="collapsed",
        key="admin_section",
    )
    st.divider()
    if st.button("🚪 Sair", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()
    st.divider()
    st.caption("Chatbot EBP — UNIFEI")

if _has_pending_changes():
    st.warning(
        "**Alterações pendentes** — a base vetorial está desatualizada. "
        "Vá em **🔄 Atualizar Base** e clique em **Atualização incremental** quando terminar todas as edições.",
    )

if section == "📄 Documentos":
    list_documents()
elif section == "📤 Upload":
    upload_section()
elif section == "✏️ Editar TXT":
    edit_documents_section()
elif section == "🔄 Atualizar Base":
    rebuild_section()
elif section == "⚙️ Config PPC":
    ppc_settings_section()
elif section == "💡 Dicas":
    tips_section()

elif section == "ℹ️ Sobre":
    st.subheader("ℹ️ Sobre o Painel Administrativo")
    st.markdown(
        "Este painel permite gerenciar a base de conhecimento do **Chatbot EBP**, "
        "o assistente virtual do curso de Engenharia de Bioprocessos da UNIFEI."
    )

    st.divider()
    st.markdown("### Como o chatbot aprende?")
    st.markdown(
        "O chatbot responde perguntas com base em documentos cadastrados aqui: "
        "PDFs e arquivos TXT organizados por categoria (normas, coordenação, estágio, etc.). "
        "Esses documentos são processados e armazenados em um **banco vetorial**, "
        "que permite ao chatbot encontrar as informações mais relevantes para cada pergunta."
    )

    st.divider()
    st.markdown("### ⚡ Atualização incremental × 🔨 Reconstrução completa")
    st.markdown("**Quando usar cada opção na aba 🔄 Atualizar Base:**")

    col1, col2 = st.columns(2)

    with col1:
        st.success("**⚡ Atualização incremental** — use na maioria dos casos")
        st.markdown(
            """
- Adicionou, editou ou removeu um ou mais documentos
- Quer atualizar o chatbot após qualquer mudança no dia a dia
- **Processa apenas o que mudou** — muito mais rápido e econômico
- Detecta automaticamente quais arquivos foram alterados
            """
        )

    with col2:
        st.error("**🔨 Reconstrução completa** — use só quando necessário")
        st.markdown(
            """
- O banco vetorial está corrompido ou com dados inconsistentes
- Você quer começar do zero por alguma razão técnica
- **Reprocessa todos os documentos** — lento e consome muito mais tokens da API
- Na dúvida, prefira sempre a atualização incremental
            """
        )

    st.info(
        "**Resumo prático:** editou arquivos? Use **Atualização incremental**. "
        "Simples assim. A reconstrução completa existe apenas para emergências."
    )

    st.divider()
    st.markdown("### Fluxo recomendado para atualizar o conteúdo")
    st.markdown(
        """
1. Faça todas as alterações que precisar (upload de PDFs, edição de TXTs, criação de arquivos)
2. Quando terminar, vá na aba **🔄 Atualizar Base**
3. Clique em **⚡ Atualização incremental**
4. Pronto — o chatbot já passa a usar as informações atualizadas

> Não precisa atualizar a base a cada arquivo salvo. Faça todas as edições primeiro e atualize uma única vez no final.
        """
    )

    st.divider()
    st.markdown("### Descrição das seções")
    st.markdown(
        """
| Seção | Função |
|-------|--------|
| 📄 Documentos | Lista e remove arquivos da base de conhecimento |
| 📤 Upload | Adiciona PDFs ou TXTs organizados por categoria |
| ✏️ Editar TXT | Edita ou cria arquivos de texto diretamente no painel |
| 🔄 Atualizar Base | Atualiza o banco vetorial após mudanças nos documentos |
| ⚙️ Config PPC | Define o link e o índice de seções do Projeto Pedagógico do Curso |
| 💡 Dicas | Gerencia avisos que aparecem na barra lateral do chatbot |
        """
    )
