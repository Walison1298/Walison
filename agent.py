from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from eda_tools import (
    carregar_csv,
    estatisticas_descritivas,
    valores_nulos,
    gerar_histograma,
    gerar_mapa_correlacao,
    executar_kmeans,
    metodo_elbow,
    gerar_boxplot,
    gerar_dispersao,
    detectar_outliers,
)

tools = [
    carregar_csv,
    estatisticas_descritivas,
    valores_nulos,
    gerar_histograma,
    gerar_mapa_correlacao,
    executar_kmeans,
    metodo_elbow,
    gerar_boxplot,
    gerar_dispersao,
    detectar_outliers,
]

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.2
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agente = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False
)

def executar_agente(prompt: str) -> str:
    try:
        return agente.run(prompt)
    except Exception as e:
        return f"❌ Erro ao executar o agente: {e}"
