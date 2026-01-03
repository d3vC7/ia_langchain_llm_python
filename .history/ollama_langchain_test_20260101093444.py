from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat = ChatOllama(
    model = "gpt-oss:20b",
    base_url = "http://localhost:11434"
)