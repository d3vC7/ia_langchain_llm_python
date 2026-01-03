from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat = ChatOllama(
    #model = "gpt-oss:20b",
    model = "llama3.2",
    base_url = "http://localhost:11434"
)

resultado = chat.invoke(
    [HumanMessage(content="¿Puedes decirme donde se encuentra Cáceres?")]
)

print(resultado)