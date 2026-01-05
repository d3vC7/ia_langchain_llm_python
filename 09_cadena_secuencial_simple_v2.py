from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature = 0
)

prompt1 = ChatPromptTemplate.from_template("Dame un simple resumen con un listado de puntos para un post de un blog acerca de {tema}")
prompt2 = ChatPromptTemplate.from_template("Escribe un post completo usando este resumen: {resumen}")

#create a simple sequencial flow without the legacy class
chain = (
    prompt1
    | llm
    | StrOutputParser()
    | (lambda x: {"resumen": x}) # Prepare input for next step
    | prompt2 
    | llm
)
result = chain.invoke({"tema": "Inteligencia artificual"})
print(result)

result2 = prompt1 | prompt1
print(result2)