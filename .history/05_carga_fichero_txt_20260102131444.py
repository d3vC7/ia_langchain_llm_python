from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
text_splitter= RecursiveCharacterTextSplitter(separador="\n", chunk_size=1000)

texts = text_splitter.create_documents([texto_completo])

print(type(texts))
print("\n\n\n")
print(type(texts[0]))
print("\n\n\n")
print(texts[0])
