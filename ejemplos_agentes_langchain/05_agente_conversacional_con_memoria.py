from langchain.tools import tool
from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import ChatOllama
from datetime import datetime
from langchain_classic.memory import ConversationBufferMemory

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature=0
)

# ejemplo sin funcion personalizada.
if __name__ == "__main__":
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    tools = load_tools(["wikipedia","llm-math",], llm=llm) 
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True )
    print(agent.invoke("Dime 5 productos esenciales para el mantenimiento del vehículo"))
    print(agent.invoke("¿Cuales de los anteriores es el mas importante?"))
    print(agent.invoke("Necsito la respuesta anterior en castellano"))