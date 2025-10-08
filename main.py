import streamlit as st
from agent import executar_agente
import os

# Load API key from secrets
if "general" in st.secrets and "GOOGLE_API_KEY" in st.secrets["general"]:
    os.environ["GOOGLE_API_KEY"] = st.secrets["general"]["GOOGLE_API_KEY"]
else:
    st.warning("⚠️ GOOGLE_API_KEY not found in st.secrets. The app may not work without it.")

st.set_page_config(page_title="Agente EDA Gemini - WSC", layout="centered")
st.title("🤖 Agente Autônomo EDA - WSC (CSV)")
st.markdown("Este agente realiza **análise exploratória de dados (EDA)** com **Gemini + LangChain**.")

st.subheader("📂 Carregar Dataset (opcional)")
uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])
if uploaded_file:
    with open("/tmp/dataset.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Dataset carregado com sucesso!")
else:
    st.info("💡 Se não enviar um arquivo, o agente usará um dataset interno padrão.")

st.subheader("💬 Converse com o Agente")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.text_input("Digite um comando (ex: estatisticas_descritivas, detectar_outliers, gerar_relatorio_completo)")
if st.button("🚀 Enviar") and prompt:
    with st.spinner("🧠 Processando..."):
        resposta = executar_agente(prompt)
        st.session_state.chat_history.append(("Você", prompt))
        st.session_state.chat_history.append(("Agente", resposta))

if st.session_state.get("chat_history"):
    st.subheader("📜 Histórico de Conversa")
    for remetente, mensagem in st.session_state.chat_history:
        st.markdown(f"**{remetente}:** {mensagem}")

st.subheader("📄 Gerar Relatório Completo")
if st.button("📝 Gerar Relatório PDF"):
    with st.spinner("Gerando relatório..."):
        resposta = executar_agente("gerar_relatorio_completo")
        st.success("✅ Relatório gerado com sucesso!")
        st.write(resposta)
        caminho_pdf = "/tmp/Agentes_Autonomos_Relatorio_Atividade_Extra.pdf"
        if os.path.exists(caminho_pdf):
            with open(caminho_pdf, "rb") as f:
                st.download_button(
                    label="📥 Baixar Relatório PDF",
                    data=f,
                    file_name="Agentes_Autonomos_Relatorio_Atividade_Extra.pdf",
                    mime="application/pdf"
                )
