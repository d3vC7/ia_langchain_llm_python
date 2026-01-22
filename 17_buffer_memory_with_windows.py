from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_ollama import ChatOllama, OllamaEmbeddings


if __name__ == "__main__":
    # Asumiendo que 'llm' ya está definido
    llm = ChatOllama( model = "llama3.2",  base_url = "http://localhost:11434" )

    # k=2 significa que son dos iteracinoes entre ia y humano.
    memory = ConversationSummaryBufferMemory(llm= llm)

    plan_viaje= '''Este fin de semana me voy de vacaciones a la playa, estaba peep snando algo que fuera bastante moderno y entretenido a la vez
        genera un plan detallado por días con que hacer en familia, extiendete todo lo que puedas'''
    
    # creamos una nueva conversación con un buffer de memoria resumida
    memory = ConversationSummaryBufferMemory(llm= llm, max_token_limit=100)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    # ejemplo con runnableWithMesageHistory
    
    print(conversation.predict(input=plan_viaje))

    # ahora veamos el resumen que nos ha hecho
    print(memory.load_memory_variables({}))

    #ver el buffer de memoria
    print(memory.buffer)