from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature=0
)

if __name__ == "__main__":
    # Definimos las erramientas a las que tendrá acceso el agente (aparte del propi motor. llm)
    tools = load_tools(["llm-math",], llm=llm) # lista de herramientas distponiles

    # inicializazmos el agente
    # 1. vemos los difentestipos de agentes a. usar
    #dir(AgentType)
    #agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
    #print(agent.run("Dime cuánto es 1598 multiplicado por 1983 y después sumas 1000"))

    # alternativa agente con create_react_agent
    template = '''Responde lo mejor que puedas usando tu conocimiento como LLM o bien las siguientes herramientas:
    {tools}
    Utiliza el siguiente formato:
    Pregunta: la pregunta de entrada que debes responder
    Pensamiento: siempre debes pensar en qué hacer
    Acción: la acción a realizar debe ser una de [{tool_names}]
    Entrada de acción: la entrada a la acción
    Observación: el resultado de la acción
    ... (este Pensamiento/Acción/Introducción de Acción/Observación puede repetirse N veces,si no consigues el resultado tras 5 intentos, para la ejecución)
    Pensamiento: ahora sé la respuesta final
    Respuesta final: la respuesta final a la pregunta de entrada original
    ¡Comenzar! Recuerda que no siempre es necesario usar las herramientas
    Pregunta: {input}
    Pensamiento:{agent_scratchpad}'''
    #agent_scratchpad: El agente no llama a una herramienta solo una vez para obtener la respuesta deseada, sino que tiene una estructura que llama a las herramientas repetidamente hasta obtener la respuesta deseada. Cada vez que llama a una herramienta, en este campo se almacena cómo fue la llamada anterior, información sobre la llamada anterior y el resultado."
   
    prompt = PromptTemplate.from_template(template)
    agente = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent = agente,
        tools = tools, 
        verbose = True,
        return_intermediate_steps = True,
        handle_parsing_errors = True
    )
    respuesta = agent_executor.invoke({"input": "Dime cuánto es 1598 multiplicado por 1983"})
    print(respuesta)