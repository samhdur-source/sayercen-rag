import os
import glob
import google.generativeai as genai
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings

SYSTEM_PROMPT = """Eres el Asistente Tecnico de SAYERCEN-D, empresa especializada en biodigestores y PTAR.
Responde UNICAMENTE con base en el siguiente contexto. Si no esta en el contexto, di que no tienes esa informacion.
Responde siempre en espanol, de forma tecnica pero clara.

Contexto: {context}

Pregunta: {question}

Respuesta:"""

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def embed_documents(self, texts):
        result = []
        for text in texts:
            r = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
            result.append(r["embedding"])
        return result

    def embed_query(self, text):
        r = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_query")
        return r["embedding"]

def cargar_documentos(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"La carpeta '{docs_path}' no existe.")
    pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))
    if not pdf_files:
        raise ValueError(f"No se encontraron PDFs en '{docs_path}'.")
    documentos = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            documentos.extend(loader.load())
        except Exception as e:
            print(f"Error cargando {pdf_path}: {e}")
    return documentos

def crear_chunks(documentos, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documentos)

def inicializar_rag(api_key, docs_path="docs"):
    documentos = cargar_documentos(docs_path)
    chunks = crear_chunks(documentos)
    embeddings = GeminiEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.2)
    prompt = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_PROMPT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    nombres = list({os.path.basename(d.metadata.get("source", "")) for d in documentos})
    return chain, len(documentos), nombres

def consultar(cadena, pregunta):
    if not pregunta.strip():
        return {"answer": "Por favor escribe una pregunta.", "source_documents": []}
    try:
        respuesta = cadena.invoke(pregunta)
        return {"answer": respuesta, "source_documents": []}
    except Exception as e:
        return {"answer": f"Error: {e}", "source_documents": []}
