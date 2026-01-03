from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)

# Initialize with your local model (e.g., nomic-embed-text)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434" # Default Ollama port
)

# carga de documento y split.
loader = TextLoader('Fuentes datos/Historia España.txt', encoding="utf-8")
documents = loader.load()

# dividir documentos en chunks, otrométodo de split 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)
#print(docs)

function_embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434" # Default Ollama port
)

# alternativa con sklearn vector. store
persist_path ="./ejemplosk_embedding_db"

# creamos la bd de vectores a partir de los documentos y la funcion embedings
vector_store = SKLearnVectorStore.from_documents(
    documents = docs,
    embedding = function_embedding,
    persist_path = persist_path,
    serializer = "parquet", #serializador o formato de la bd y lo definomos como parquet.
)

# fuerza a guardar los nuevos embeddings. en el disco.
vector_store.persist()

# creamos un nuevo documentoque será nuestra consulta para buscar. el de mayor similitud en la bd
consulta = "dame informaciónde la primera guerra mundial"
docs = vector_store.similarity_search(consulta)
#print(docs[0].page_content)

# cargar la bd de vectores (uso. posterior una vez ya creada)
vector_store_connection = SKLearnVectorStore(
    embedding = function_embedding,
    persist_path = persist_path,
    serializer = "parquet"
)
print("una instancia de la bd se ha cargao desde " , persist_path)

type(vector_store_connection)