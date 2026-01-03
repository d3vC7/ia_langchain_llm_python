from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import WikipediaLoader

chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)

# cargar archivos desde wirkipedia
def responder_wikipedia(persona, pregunta_arg):
    docs = WikipediaLoader(query=persona, lang="es", load_max_docs=10)
    contexto_extra = docs.load()[0].page_content # para que sea más rapido el pasamos el primer documento

    # pregunta el usuario
    human_prompt =HumanMessagePromptTemplate.from_template('esponde a esta pregunta \n{pregunta}, aquí tienes el contenido extra:\n{contenido}')

    # construir el promt
    chat_prompt =ChatPromptTemplate.from_messages([human_prompt])

    