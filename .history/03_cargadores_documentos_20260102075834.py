from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)