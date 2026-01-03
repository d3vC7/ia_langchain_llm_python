from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.text_splitter import CharacterTextSplitter

chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)

# carga de fichero
with open('Fuentes datos/Historia España.txt',encoding="utf") as file:
    texto_completo = file.read()

# número de caracteres
print(len(texto_completo))

# número de palabras
print(len(texto_completo.split()))


# Transformador "Characer Text Splitter"

