from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import CSVLoader


chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)

