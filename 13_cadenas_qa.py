from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import WikipediaLoader
from langchain_classic.chains import SimpleSequentialChain, LLMChain,TransformChain


from langchain_community.vectorstores import SKLearnVectorStore


if __name__ == "__main__":
    # Asumiendo que 'llm' ya está definido
    llm = ChatOllama( model = "llama3.2",  base_url = "http://localhost:11434" )
    function_embedding = OllamaEmbeddings( model="nomic-embed-text", base_url="http://localhost:11434" )

    vector_store_connection = SKLearnVectorStore(embedding=function_embedding, 
                                                 persist_path="ejemplosk_embedding_db", 
                                                 serializer="parquet")
    # cargar. cadena qa
    from langchain_classic.chains.question_answering import load_qa_chain
    from langchain_classic.chains.qa_with_sources import load_qa_with_sources_chain

    chain = load_qa_chain(llm,chain_type='stuff') 
    #chain_type='stuff' se usa cuando se desea una manera simple y directa de cargar y procesar el contenido completo 
    # sin dividirlo en fragmentos más pequeños. Es ideal para situaciones donde el volumen de datos no es demasiado 
    # grande y se puede manejar de manera eficiente por el modelo de lenguaje en una sola operación.
    question = "¿Qué pasó en el siglo de Oro?"
    docs = vector_store_connection.similarity_search(question)
    print(chain.run(input_documents=docs,question=question))