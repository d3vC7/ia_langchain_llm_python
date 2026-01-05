from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains import LLMChain 
from langchain_classic.chains.sequential import SimpleSequentialChain

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature = 0
)

template = "Dame un simple resumen con un listado de puntos para un post de un blog acerca de {tema}"
prompt1 = ChatPromptTemplate.from_template(template)
chain_1 = LLMChain(llm=llm,prompt=prompt1)

template = "Escribe un post completo usando este resumen: {resumen}"
prompt2 = ChatPromptTemplate.from_template(template)
chain_2 = LLMChain(llm=llm,prompt=prompt2)

full_chain = SimpleSequentialChain(chains=[chain_1,chain_2],
                                  verbose=True) #verbose=True nos irá dando paso a paso lo que hace, pudiendo ver los resultados intermedios

#full_chain = prompt1 | llm | prompt2 | llm
result = full_chain.invoke(input="Inteligencia Artificial")
print(result)