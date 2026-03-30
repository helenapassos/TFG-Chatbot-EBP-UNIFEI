"""
Ponto de entrada do app multi-page.
Edite os títulos (title) abaixo para mudar os nomes na barra lateral.
"""

import streamlit as st

st.set_page_config(
    page_title="Chatbot EBP — UNIFEI",
    page_icon="🎓",
    layout="centered",
)

# ── Edite aqui os nomes e ícones que aparecem na sidebar ──
chatbot_page = st.Page(
    "pages/0_Chatbot.py",
    title="Chatbot EBP",
    icon="🎓",
    default=True,
)

admin_page = st.Page(
    "pages/1_Painel_Administrativo.py",
    title="Painel Administrativo",
    icon="🔐",
)

pg = st.navigation([chatbot_page, admin_page])
pg.run()
