from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories  import ChatMessageHistory


if __name__ == "__main__":
    # Asumiendo que 'llm' ya está definido
    chat = ChatOllama( model = "llama3.2",  base_url = "http://localhost:11434" )
    history = ChatMessageHistory()
    consulta = "Hola, ¿cómo estás? Necesito ayuda para reconfiugrar el router."
    history.add_user_message(consulta)
    resultado = chat.invoke([HumanMessage(content=consulta)])
    history.add_ai_message(resultado.content)
    print(history)
    print("\n\n\n")
    print(history.messages) 