from langchain.tools import tool
from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature=0
)

@tool
def persona_amable(text: str) -> str:
    '''Retorna la persona más amable. Se espera que la entrada esté vacía ""
    y retorna la persona más amable del universo'''
    return "Miguel Celebres"

# definimos. las herramisnt qeu tednrá acceso el agente cuando ejecutamos

# ejemplo sin funcion personalizada.
if __name__ == "__main__":
    tools = load_tools(["wikipedia", "llm-math", ], llm = llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True )
    agent.invoke("¿Quién es la persona más amable del universo?")