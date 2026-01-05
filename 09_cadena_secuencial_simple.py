from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature = 0
)

# creación del objeto LlmChain
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "Dame sólo un nombre de compañía que sea simpático para una compañía que fabrique {producto}"
)
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
# oldway 
# chain = LLMChain(llm=chat, prompt = chat_prompt_template)
#print(chain.invoke(input = "Lavadoras"))
chain = chat_prompt_template | chat
print(chain.invoke(input = "Lavadoras"))
print(chain.invoke(input = "Lavadoras").content)

# ahora cadena secuencia simpleç
