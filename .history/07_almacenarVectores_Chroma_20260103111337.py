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