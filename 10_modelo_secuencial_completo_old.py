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

template1 = "Dame un resumen del rendimiento de este trabajador \n {revision_rendimiento}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain_1 = LLMChain(llm=llm, prompt=prompt1, output_key = "resumen_revision")
print(chain_1)

template2 = "Identifica las debilidades de este trabajador dentro de este resumen de la revisión \n {resumen_revision}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain_2 = LLMChain(llm=llm, prompt=prompt2, output_key = "debilidades")
print(chain_2)

template3 = "Crea un plan de mejora para ayudar en estas debilidades \n {debilidades}"
prompt3 = ChatPromptTemplate.from_template(template3)
chain_3 = LLMChain(llm=llm, prompt=prompt2, output_key = "plan_mejora")


# old way
seq_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3],
    input_variables=["revision_rendimiento"],
    output_variables=["resumen_revision", "debilidades", "plan_mejora"],
    verbose=True)

revision_rendimiento_empleado = '''
Revisión de Rendimiento del Empleado

Nombre del Empleado: Juan Pérez
Posición: Analista de Datos
Período Evaluado: Enero 2023 - Junio 2023

Fortalezas:
Juan ha demostrado un fuerte dominio de las herramientas analíticas y ha proporcionado informes detallados y precisos que han sido de gran ayuda para la toma de decisiones estratégicas. Su capacidad para trabajar en equipo y su disposición para ayudar a los demás también han sido notables. Además, ha mostrado una gran ética de trabajo y una actitud positiva en el entorno laboral.

Debilidades:
A pesar de sus muchas fortalezas, Juan ha mostrado áreas que necesitan mejoras. En particular, se ha observado que a veces tiene dificultades para manejar múltiples tareas simultáneamente, lo que resulta en retrasos en la entrega de proyectos. También ha habido ocasiones en las que la calidad del trabajo ha disminuido bajo presión. Además, se ha identificado una necesidad de mejorar sus habilidades de comunicación, especialmente en lo que respecta a la presentación de datos complejos de manera clara y concisa a los miembros no técnicos del equipo. Finalmente, se ha notado una falta de proactividad en la búsqueda de soluciones a problemas imprevistos, confiando a menudo en la orientación de sus superiores en lugar de tomar la iniciativa.
'''

results = seq_chain.invoke(revision_rendimiento_empleado)

print(results)

#Resultado final
print(results['plan_mejora'])

#Se puede accceder a los resultados intermedios:
print(results["debilidades"])