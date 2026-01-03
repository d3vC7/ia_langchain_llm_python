from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_chroma import Chroma

chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)


function_embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434" # Default Ollama port
)

# carga de documento y split.
loader = TextLoader('Fuentes datos/Historia España.txt', encoding="utf-8")
documents = loader.load()

# dividir documentos en chunks, otrométodo de split 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

# cargar en chromaDB
db = Chroma.from_documents(docs, function_embedding, collection_name="langchain",
                           persist_directory='./ejemplo_embedding_db')

# cargar los embeddings desde el disco creando la conexióna chromaDB
db_connection = Chroma(persist_directory='./ejemplo_embedding_db', embedding_function=function_embedding)

# creamos un nuevo. documento para buscar el de mayor similituden nuetra base. de datos de vectores
nuevo_documento = "what did FDR say about chile country?"

docs = db_connection.similarity_search(nuevo_documento)

# el primer elemento es el de mayor similitud
print(docs[0].page_content)


# como añadir nueva informacion. a la bd de cetores.

# cargar nuevo documetno y dividirlo
loader_nuevo_documento = TextLoader('Fuentes datos/Nuevo_documento.txt', encoding="utf8")
documents = loader_nuevo_documento.load()

text_splitter_nuevo_documento =