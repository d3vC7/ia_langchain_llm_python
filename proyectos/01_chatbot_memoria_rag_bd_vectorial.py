from langchain.tools import tool
from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import ChatOllama
from datetime import datetime
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature=0
)

function_embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434" # Default Ollama port
)

persist_path = "./proyecto_chatbot_bd."

# ejemplo sin funcion personalizada.
if __name__ == "__main__":
    memory =  ConversationBufferMemory(memory_key="chat_history")
    persist_pat = "./ejemplosk_embedding_db"
    vector_store_connection = SKLearnVectorStore(
        embedding = function_embedding,
        persist_path = persist_path,
        serializer = "parquet"
    )
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor = compressor,
                                                    base_retriever = vector_store_connection.as_retriever())
    
    # creamos una nueva herramienta a partir de la bd vectorial para obtener resultados optimizados
    from langchain.tools import tool

    @tool
    def consulta_interna(text: str) -> str:
        '''Retorna respuestas sobre la historia de España. Se espera que la entrada sea una cadena de texto y
        retorna una cadena con el resultado más relevante. Si la respuesta con esta herramienta es relevante no debes
        usar ninguna herramienta más ni tu propio conocimiento como LLM'''
        compressed_docs = compression_retriever.invoke(text)
        resultado = compressed_docs[0].page_content
        return resultado
    
    #tools = load_tools(["wikipedia","llm-math",], llm=llm) 
    tools = load_tools(["wikipedia","llm-math",], llm=llm) 
    tools = tools + [consulta_interna]

    agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True )
    
    print(agent.invoke("¿Qué periodo abarca cronologicamente en España el sigo de oro?"))
    print(agent.invoke("¿Qué pasó durante la misma etapa. en Francia?"))
    print(agent.invoke("¿Qué periodo abarca cronologicamente en España el sigo de oro?"))