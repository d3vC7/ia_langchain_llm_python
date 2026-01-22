from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_ollama import ChatOllama, OllamaEmbeddings


if __name__ == "__main__":
    # Asumiendo que 'llm' ya está definido
    chat = ChatOllama( model = "llama3.2",  base_url = "http://localhost:11434" )

    # k=2 significa que son dos iteracinoes entre ia y humano.
    memory = ConversationBufferWindowMemory(k=2)

    # creamos una instancia a la cadena conversacional con el LLM y el objeto de memoria
    conversation = ConversationChain(llm = chat, memory = memory, verbose= True)
    # Ejemplo con RunnableWithMessageHistory

    print(conversation.predict(input="Hola, ¿cómo estás?"))

    print(conversation.predict(input="Necesito un consejo para tener un gran día."))

    print("\n\n ahora imprimiremos la memoria y debe tener un limite de 2 interacciones \n\n")
    print(memory.buffer)
    
    
    
    
    
    
    
    
    
    #Conectar una conversación a la memoria.
    # creamos una instancia de la cadena conversacional con el llm y un objeto de la memoria
    conversation = ConversationChain(llm=chat, memory = memory, verbose=True)

    # lanzamos el primer prompt (human message)
    print(conversation.predict(input="Hola, necesito saber cómo usar mis datos históricos para crear un bot de preguntas y respuestas"))
    
    print(conversation.predict(input="Necesito más detalle de cómo implementarlo"))
    
    # obtenems el histórico
    #print(memory.buffer)

    print("\n\n\n")

    # cargamos la variable de memoria
    #print(memory.load_memory_variables({}))


    # guardar y cargar la memoría para un posterior uso
    print("\n\n guardar y cargar la memoría para un posterior uso \n")
    #print(conversation.memory)
   

    # la forma mas sensilla para guardar esta info es con la libreria pickle 
    print("\n\n imprimir valor de pickle \n")
    import pickle
    pickled_str = pickle.dumps(conversation.memory)

    print(pickled_str)
    # escribiendo un objeto binario
    with open ('memory.pkl', 'wb') as f:
        f.write(pickled_str)

    #leer el objeto binario
    memoria_cargada = open('memory.pkl', 'rb').read() 

    #creamos una nueva instancia LLm para asegurar eque está todo ok.
    conversacion_recargada = ConversationChain(
        llm = chat,
        memory = pickle.loads(memoria_cargada),
        verbose = True
    )

    print("\n\n tenemos la conversa guardada y recargada \n")
    print(conversacion_recargada.memory.buffer)