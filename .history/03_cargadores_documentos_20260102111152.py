from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import CSVLoader, BSHTMLLoader, PyPDFLoader


chat = ChatOllama(
    model = "llama3.2",
    base_url = "http://localhost:11434"
)

# cargamos el fichero csv
loader = CSVLoader('Fuentes datos/datos_ventas_small.csv', csv_args={'delimiter':';'})
data = loader.load()
#print(type(data))

#print(data[0])
#print(data[1])

# cargamos el html
print("\n\n\n")
loader_html =BSHTMLLoader('Fuentes datos/ejemplo_web.html')
data_html =loader_html.load()
#print(data_html[0].page_content)

loader_pdf = PyPDFLoader('Fuentes datos/documentopdf.pdf')
pages=loader_pdf.load_and_split()
print(type(pages))
print(pages[0])

print(pages[0].page_content)

human_template= '"Necesito que hagas un resumen del siguiente texto: \n  {contenido}"'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
chat_prompt.format_prompt(contenido = pages[0].page_content)