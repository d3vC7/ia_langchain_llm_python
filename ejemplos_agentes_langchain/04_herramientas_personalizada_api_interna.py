from langchain.tools import tool
from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import ChatOllama
from datetime import datetime

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature=0
)

# caso de uso 2, api interna
@tool
def nombre_api_interna(text: str) -> str:
    '''Conecta a la API_xxx que realiza la tarea xx, debes usar esta api key'''
    # definir coneccióna la api interna y devolver un resultado
    return "salida_resultado"

# caso de uso 3, consultar la hora actual. Crear una herramienta personalizad para esto
@tool
def hora_actual(text: str)->str:
    '''Retorna la hora actual, debes usar esta función para cualquier consulta sobre la hora actual. Para fechas que no sean
    la hora actual, debes usar otra herramienta. La entrada está vacía y la salida retorna una string'''
    return str(datetime.now())


# ejemplo sin funcion personalizada.
if __name__ == "__main__":
    tools = load_tools(["wikipedia", "llm-math", ], llm = llm)
    
    tools = tools + [nombre_api_interna] + [hora_actual]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True )
    agent.invoke("¿Cual es la hora actual?")
