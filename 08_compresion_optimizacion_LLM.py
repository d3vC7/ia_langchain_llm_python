
from langchain_community.document_loaders import WikipediaLoader, TextLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

class CompresionOptimizada:

    def __init__(self, nombre):
        self.nombre = nombre

    def carga_documento(self):
        # 1. carga documentos desde wikipedia.
        loader = WikipediaLoader(query="Lenguaje Python", lang="es", load_max_docs=7)
        documents = loader.load()
        print(documents)
        print(len(documents))
        # paso 2 vamos a dividr los documents en fragmentos
        text_splitter =RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
        docs =text_splitter.split_documents(documents)
        print(len(docs))
        
        function_embedding = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434" # Default Ollama port
        )
        persist_path = "./ejemplo_wiki_bd_v2"
        vector_store = SKLearnVectorStore.from_documents(
            documents = docs, 
            embedding = function_embedding,
            persist_path = persist_path,
            serializer = "parquet" # el serializador o formato de la bd lo definimos. como  parquet
        )
        vector_store.persist()
        consulta = "¿Por qué el lenguaje de python se llama así?"
        docs = vector_store.similarity_search(consulta)
        print("\n\n\n")
        print("primero de los documentos a la consulta sin LLM:::\n ")
        print(docs[0].page_content)
        print("\n\n\n")

        llm = ChatOllama(
            model = "llama3.2",
            base_url = "http://localhost:11434",
            temperature = 0
        )
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor = compressor,
                                                            base_retriever = vector_store.as_retriever())
        compressed_docs = compression_retriever.invoke("¿Por qué el lenguaje python se llama así?")
        print("\n\n usando compression, la respuesta es::: \n\n")
        print(compressed_docs[0].page_content)
        return
    
    def parte2():
        # paso 3. vamos a conectarcon los embedigns
        function_embedding = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434" # Default Ollama port
        )

        # paso 4, incrustar los documentso en bd vectores
        persist_path = "./ejemplo_wiki_bd"
        vector_store = SKLearnVectorStore.from_documents(
            documents = docs, 
            embedding = function_embedding,
            persist_path = persist_path,
            serializer = "parquet" # el serializador o formato de la bd lo definimos. como  parquet
        )
        # fuerzaz a gauradr los cambios embeddings, no necesario en chroma
        vector_store.persist()

        # 5. consultar normal similitud coseno
        # creamos un nuevo docuemntoqeu será nuestra consulta.
        consulta = "¿Por qué el lenguaje de python se llama así?"
        docs = vector_store.similarity_search(consulta)
        print(docs[0].page_content)

        # Paso 6,  vamos a hacer. una consulta con compresión. contextual usando LLMS
        llm = ChatOllama(
            model = "llama3.2",
            base_url = "http://localhost:11434",
            temperature = 0
        )
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor = compressor,
                                                            base_retriever = vector_store.as_retriever())
        compressed_docs = compression_retriever.invoke("¿Por qué el lenguaje python se llama así?")
        print("\n\n usando compression \n\n")
        print(compressed_docs[0].page_content)




if __name__ == '__main__':
    print("--- Ejecución principal del script ---")
    objeto = CompresionOptimizada("Python")
    objeto.carga_documento()