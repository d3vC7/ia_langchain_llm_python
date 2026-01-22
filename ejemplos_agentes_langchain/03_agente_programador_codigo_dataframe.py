from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
import pandas as pd


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

    df = pd.read_excel('datos_ventas_small.xlsx')
    print(df.head())

    #print(agent.invoke(f'''¿Qué sentencias de código tendría que ejecutar para obtener la suma de venta total agregada por línea de producto?
    #                   Este sería el dataframe {df}, no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar'''))
    
    print(df.groupby('Línea Producto')['Venta total'].sum())

    print("\n\n\n\n")

    #print(agent.invoke(f'''Cuál es la suma agregada de la venta total para la línea de producto \"Motorcycles\"? Este sería el dataframe {df} '''))

    print(agent.invoke(f'''¿Qué sentencias de código tendría que ejecutar para tener una visualización con la librería Seaborn que agregue a nivel de Línea de Producto el total de venta? Este sería el dataframe {df}, recuerda que no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar'''))