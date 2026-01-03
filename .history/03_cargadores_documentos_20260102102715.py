from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import CSVLoader


chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)

# cargamos el fichero csv
loader = CSVLoader('Fuentes datos/datos_ventas_small.csv', csv_args={'delimeter':';'})
data = loader.load()
type(data)