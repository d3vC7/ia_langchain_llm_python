from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama, OllamaEmbeddings


if __name__ == "__main__":
    # Asumiendo que 'llm' ya está definido
    chat = ChatOllama( model = "llama3.2",  base_url = "http://localhost:11434" )
    memory = ConversationBufferMemory()

    #Conectar una conversación a la memoria
    conversation = ConversationChain(llm=chat, memory = memory,verbose=True)
    print(conversation.predict(input="Hola, necesito saber cómo usar mis datos históricos para crear un bot de preguntas y respuestas"))
    
    print(conversation.predict(input="Necesito más detalle de cómo implementarlo"))

    print(memory.buffer)

    print(memory.load_memory_variables({}))

    import pickle
    pickled_str = pickle.dumps(conversation.memory) # objeto binario para guardar info
    with open ('memory.pkl', 'wb') as f:
        f.write(pickled_str)

    memoria_cargada = open('memory.pkl', 'rb').read() # leemos objeto binario

    conversacion_recargada = ConversationChain(
        llm = chat,
        memory = pickle.loads(memoria_cargada),
        verbose = True
    )

    print(conversacion_recargada.memory.buffer)