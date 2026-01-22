from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool


import os

# se usará serpapi como motor de búsqueda
# este caso funciona mucho mejor con open ai, acá con el modelo de

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature=0
)

if __name__ == "__main__":

    os.environ["SERPAPI_API_KEY"] = "64a5ad8cfe3ce7629215dfcab0884b328a266d914991df173ba8d1b5b65b0ec5"

    # creamos. el. agente. para crear y ejecutar código python.
    agent = create_python_agent(tool = PythonREPLTool(),
                                llm= llm, 
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    lista_ejemplo = [3,1,5,3,5,6,7,35,10]

    print(agent.invoke(f'''ordena la lista {lista_ejemplo}'''))

