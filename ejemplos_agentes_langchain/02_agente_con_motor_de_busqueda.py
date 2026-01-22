from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
import os

# se usará serpapi como motor de búsqueda
# este caso funciona mucho mejor con open ai, acá con el modelo de

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature=0
)

if __name__ == "__main__":
    #serp_api_key = "Xm358nRA2Wwuh2pp1P9gy4vQ"
    os.environ["SERPAPI_API_KEY"] = "64a5ad8cfe3ce7629215dfcab0884b328a266d914991df173ba8d1b5b65b0ec5"

    # definimoslas. herramintas. a las qeu tendrá acceso el agente
    tools = load_tools(["serpapi", "llm-math", ], llm = llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True )
    agent.invoke("El año de nacimiento de Albert Einsten multiplicalo por 3?")


