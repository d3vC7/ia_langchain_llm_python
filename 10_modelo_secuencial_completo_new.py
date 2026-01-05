from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains import LLMChain 
from langchain_classic.chains.sequential import SimpleSequentialChain
from langchain_classic.chains import LLMChain, SequentialChain #importamos el SequentialChain que es el modelo completo


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434",
    temperature = 0
)

# Versión recomendada para tu caso específico:
def crear_analisis_rendimiento_2026(revision_rendimiento_empleado, llm):
    """
    Migración directa de tu código original a LCEL
    """
    # Crear los prompts
    prompt_resumen = ChatPromptTemplate.from_template(
        "Dame un resumen del rendimiento de este trabajador \n {revision_rendimiento}"
    )
    prompt_debilidades = ChatPromptTemplate.from_template(
        "Identifica las debilidades de este trabajador dentro de este resumen de la revisión \n {resumen_revision}"
    )
    prompt_plan = ChatPromptTemplate.from_template(
        "Crea un plan de mejora para ayudar en estas debilidades \n {debilidades}"
    )


    
    # Pipeline con LCEL - sintaxis moderna sin warnings
    pipeline = (
        # Paso 1: Generar resumen
        RunnablePassthrough.assign(
            resumen_revision=lambda x: (
                prompt_resumen 
                | llm 
                | StrOutputParser()
            ).invoke({"revision_rendimiento": x["revision_rendimiento"]})
        )
        # Paso 2: Identificar debilidades
        .assign(
            debilidades=lambda x: (
                prompt_debilidades 
                | llm 
                | StrOutputParser()
            ).invoke({"resumen_revision": x["resumen_revision"]})
        )
        # Paso 3: Crear plan de mejora
        .assign(
            plan_mejora=lambda x: (
                prompt_plan 
                | llm 
                | StrOutputParser()
            ).invoke({"debilidades": x["debilidades"]})
        )  
    )
    # Ejecutar el análisis
    resultados = pipeline.invoke({
        "revision_rendimiento": revision_rendimiento_empleado
    })

    # Ejemplo de uso con tu texto:
if __name__ == "__main__":
    # Asumiendo que 'llm' ya está definido
    # llm = ChatOpenAI(temperature=0.7, model="gpt-4")
    
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
    
    # Sin warnings, código moderno
    resultados = crear_analisis_rendimiento_2026(revision_rendimiento_empleado, llm)
    
    # Acceder a los resultados
    print("Resumen:", resultados["resumen_revision"])
    print("\nDebilidades:", resultados["debilidades"])
    print("\nPlan de mejora:", resultados["plan_mejora"])