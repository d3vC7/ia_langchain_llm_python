from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains import LLMChain
from langchain_classic.chains.sequential import SimpleSequentialChain
from langchain_classic.chains import LLMChain, SequentialChain #importamos el SequentialChain que es el modelo completo

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature = 0
)

plantilla_soporte_basico_cliente = '''Eres una persona que asiste a los clientes de automóviles con preguntas básicas que pueden
necesitar en su día a día y que explica los conceptos de una manera que sea simple de entender. Asume que no tienen conocimiento
previo. Esta es la pregunta del usuario/n{input}'''

plantilla_soporte_avanzado_mecánico = '''Eres un experto en mecánica que explicas consultas avanzadas a los mecánicos
de la plantilla. Puedes asumir que cualquier que está preguntando tiene conocimientos avanzados de mecánica. 
Esta es la pregunta del usuario/n{input}'''

#Debemos crear una lista de diccionarios, cada diccionario contiene su nombre, la descripción (en base a la cual el enrutador
#hará su trabajo) y el prompt a usar en cada caso. prompts a enrutar
prompt_infos = [
    {'name':'mecánica básica','description': 'Responde preguntas básicas de mecánicas a clientes',
     'prompt_template':plantilla_soporte_basico_cliente},
    {'name':'mecánica avanzada','description': 'Responde preguntas avanzadas de mecánica a expertos con conocimiento previo',
     'prompt_template':plantilla_soporte_avanzado_mecánico},
]

# conversation chain.
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# diccionario de bloques LLM Chain.
print (destination_chains)

#Creamos el prompt y cadena por defecto puesto que son argumento obligatorios que usaremos posteriormente
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm,prompt=default_prompt)

# ahora a crear el multiroutring. template.
#Importamos una plantilla que podremos formatear su parámetro {destinations} que tendrá cada nombre y descripción de la información de prompts
from langchain_classic.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

print(MULTI_PROMPT_ROUTER_TEMPLATE) #El parámetro importante es {destinations}, debemos formatearlo en tipo string

# DESTINOS DE ROUTING
#Creamos una string global con todos los destinos de routing usando el nombre y descripción de "prompt_infos"
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
print("\n\ndestinations_str\n")
print(destinations_str)

# router prompt
from langchain_classic.chains.router.llm_router import LLMRouterChain,RouterOutputParser

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str #Formateamos la plantilla con nuestros destinos en la string destinations_str
)
print(router_template)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), #Para transformar el objeto JSON parseándolo a una string
)

#routing chain call
from langchain_classic.chains.router import MultiPromptChain
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, #El objeto con los posibles LLMChain que creamos al inicio
                         default_chain=default_chain, verbose=True #Indicamos el LLMChain por defecto (obligatorio)
                        )
# print(chain.invoke("¿cómo cambio el aceite de mi coche?"))

print(chain.invoke("¿cómo funciona internamente un catalizador?"))