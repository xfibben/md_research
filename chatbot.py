from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
import os
import PyPDF2


# Configurar clave de API de OpenAI
os.environ["OPENAI_API_KEY"] = "api"

# 1. Cargar y dividir el texto del libro
def cargar_texto_y_dividir(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as file:
        contenido = file.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragmentos = text_splitter.split_text(contenido)
    return fragmentos


def extraer_texto_pdf(ruta_pdf):
    texto = ""
    with open(ruta_pdf, "rb") as archivo:
        lector_pdf = PyPDF2.PdfReader(archivo)
        for pagina in lector_pdf.pages:
            texto += pagina.extract_text()
    return texto


# 2. Crear o cargar el almacén vectorial
def configurar_almacen_vectorial(fragmentos, embeddings, ruta_vectorstore):
    if os.path.exists(ruta_vectorstore):
        vectorstore = FAISS.load_local(ruta_vectorstore, embeddings)
    else:
        vectorstore = FAISS.from_texts(fragmentos, embeddings)
        vectorstore.save_local(ruta_vectorstore)
    return vectorstore

# 3. Crear el pipeline RAG
def crear_chatbot(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever()
    chatbot = RetrievalQA(llm=llm, retriever=retriever)
    return chatbot

# 4. Ejecutar el chatbot
def ejecutar_chatbot(chatbot):
    print("Chatbot de Derechos del Consumidor Peruano (escribe 'salir' para terminar)")
    while True:
        consulta = input("Tú: ")
        if consulta.lower() == "salir":
            print("¡Adiós!")
            break
        respuesta = chatbot.run(consulta)
        print(f"Chatbot: {respuesta}")

# Configuración principal
if __name__ == "__main__":
    # Ruta al PDF
    ruta_pdf = "chatbot/data.pdf"
    ruta_vectorstore = "vectorstore_derechos_consumidor"

    # Extraer texto del PDF y dividirlo en fragmentos
    print("Extrayendo texto del PDF y procesando...")
    texto_extraido = extraer_texto_pdf(ruta_pdf)

    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragmentos = text_splitter.split_text(texto_extraido)

    # Configurar embeddings y almacén vectorial
    print("Configurando el almacén vectorial...")
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    embeddings = OpenAIEmbeddings()
    vectorstore = configurar_almacen_vectorial(fragmentos, embeddings, ruta_vectorstore)

    # Crear el chatbot
    print("Inicializando el chatbot...")
    chatbot = crear_chatbot(vectorstore)

    # Ejecutar el chatbot
    ejecutar_chatbot(chatbot)


