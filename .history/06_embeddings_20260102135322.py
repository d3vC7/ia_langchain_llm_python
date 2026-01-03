from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import CSVLoader
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

texto = "Esto es un texto enviado a llama para ser incrustado en un vector n-dimensional"

# el texto se ha convertido en una serie de dimensiones
embedded_text = embeddings.embed_query(texto)

print(type(embedded_text))
print(embedded_text)