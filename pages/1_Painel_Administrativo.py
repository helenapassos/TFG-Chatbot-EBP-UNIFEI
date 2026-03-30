"""
Painel Administrativo — Upload e gerenciamento da base de conhecimento.
Acesso protegido por senha para coordenadores.
"""

import shutil
import datetime

import streamlit as st

import config
from config import (
    load_ppc_config, save_ppc_config, load_tips, save_tips,
    cleanup_expired_tips, _is_tip_active,
)
from rag.ingest import ingest_all, load_vectorstore


LOGO_URL = "https://portalpadrao.ufma.br/ineof/imagens/logo-unifei-oficial.png"


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

        for uploaded_file in uploaded_files:
            dest_path = dest_dir / uploaded_file.name
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} salvo em {category}/")

        st.rerun()


def rebuild_section():
    st.subheader("🔄 Atualizar Base Vetorial")
    st.markdown(
        "Após adicionar ou remover documentos, reconstrua o banco vetorial "
        "para que o chatbot utilize as informações atualizadas."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🔨 Reconstruir banco vetorial",
            use_container_width=True,
            type="primary",
        ):
            vs_dir = config.VECTORSTORE_DIR
            if vs_dir.exists():
                shutil.rmtree(vs_dir)
                vs_dir.mkdir(parents=True, exist_ok=True)

            with st.spinner("Processando documentos... Isso pode levar alguns minutos."):
                try:
                    vs = ingest_all()
                    count = vs._collection.count()
                    st.success(
                        f"✅ Banco vetorial reconstruído com sucesso! "
                        f"{count} chunks indexados."
                    )
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Erro ao reconstruir: {e}")

    with col2:
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

    col_save, col_revert = st.columns(2)

    with col_save:
        if st.button("💾 Salvar alterações", use_container_width=True, type="primary"):
            if not edited_content.strip():
                st.error("O conteúdo não pode ficar vazio.")
            else:
                selected_path.write_text(edited_content, encoding="utf-8")
                st.success(
                    f"✅ Arquivo `{selected_path.name}` salvo com sucesso! "
                    "Lembre-se de **reconstruir a base vetorial** na aba "
                    "'🔄 Atualizar Base' para que as mudanças tenham efeito no chatbot."
                )

    with col_revert:
        if st.button("↩️ Desfazer edição", use_container_width=True):
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
    st.markdown("Utilize este painel para gerenciar os documentos da base de conhecimento do chatbot.")
    st.divider()
    if st.button("🚪 Sair"):
        st.session_state.admin_authenticated = False
        st.rerun()
    st.divider()
    st.caption("Chatbot EBP — UNIFEI")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📄 Documentos", "📤 Upload", "✏️ Editar TXT", "🔄 Atualizar Base", "⚙️ Config PPC", "💡 Dicas",
])

with tab1:
    list_documents()

with tab2:
    upload_section()

with tab3:
    edit_documents_section()

with tab4:
    rebuild_section()

with tab5:
    ppc_settings_section()

with tab6:
    tips_section()
