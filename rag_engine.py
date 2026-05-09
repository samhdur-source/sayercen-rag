"""
rag_engine.py — Motor RAG para SAYERCEN-D Technical Assistant

Responsabilidades:
- Cargar documentos PDF desde la carpeta /docs
- Dividir en chunks y crear embeddings
- Indexar en FAISS (vector store local, sin servidor)
- Responder preguntas usando Gemini como LLM

Autor: SAYERCEN-D / Proyecto Final Módulo Python para IA
"""

import os
import glob
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory`nfrom langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


# ── Prompt del sistema ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres el Asistente Técnico de SAYERCEN-D (Servicios Ambientales y de 
Energías Renovables del Centro), empresa especializada en biodigestores, plantas de 
tratamiento de aguas residuales y energías renovables con 21 años de experiencia.

Responde ÚNICAMENTE con base en los documentos técnicos proporcionados.
Si la información no está en los documentos, dilo claramente: 
"No encuentro información sobre ese tema en los documentos disponibles."

Responde siempre en español, de forma técnica pero clara.
Cuando menciones datos numéricos (volúmenes, costos, eficiencias), cítalos exactamente 
como aparecen en los documentos.

Contexto de los documentos:
{context}

Historial de conversación:
{chat_history}

Pregunta: {question}

Respuesta técnica:"""


def cargar_documentos(docs_path: str = "docs") -> list:
    """Carga todos los PDFs de la carpeta especificada.

    Args:
        docs_path: Ruta a la carpeta que contiene los PDFs.

    Returns:
        Lista de documentos LangChain cargados.

    Raises:
        FileNotFoundError: Si la carpeta no existe.
        ValueError: Si no se encuentran PDFs en la carpeta.
    """
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"La carpeta '{docs_path}' no existe.")

    pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No se encontraron archivos PDF en '{docs_path}'. "
            "Por favor, añade tus documentos PDF a esa carpeta."
        )

    documentos = []
    errores = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documentos.extend(docs)
        except Exception as e:
            errores.append(f"{os.path.basename(pdf_path)}: {e}")

    if errores:
        print(f"⚠️ Advertencia — archivos con error: {'; '.join(errores)}")

    return documentos


def crear_chunks(documentos: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Divide documentos en fragmentos para el indexado.

    Args:
        documentos: Lista de documentos LangChain.
        chunk_size: Tamaño máximo de cada chunk en caracteres.
        chunk_overlap: Solapamiento entre chunks consecutivos.

    Returns:
        Lista de chunks listos para indexar.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documentos)


def construir_vectorstore(chunks: list, api_key: str) -> FAISS:
    """Crea el índice vectorial FAISS con embeddings de Google.

    Args:
        chunks: Lista de chunks de texto.
        api_key: API Key de Google Gemini.

    Returns:
        Vector store FAISS listo para consultas.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def construir_cadena_rag(vectorstore: FAISS, api_key: str) -> ConversationalRetrievalChain:
    """Construye la cadena RAG conversacional con memoria.

    Args:
        vectorstore: Índice FAISS con los documentos.
        api_key: API Key de Google Gemini.

    Returns:
        Cadena RAG lista para recibir preguntas.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )

    memoria = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT,
    )

    cadena = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memoria,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return cadena


def inicializar_rag(api_key: str, docs_path: str = "docs") -> tuple[ConversationalRetrievalChain, int, list]:
    """Pipeline completo: carga → chunks → embeddings → cadena RAG.

    Args:
        api_key: API Key de Google Gemini.
        docs_path: Ruta a la carpeta con PDFs.

    Returns:
        Tupla con (cadena_rag, num_documentos, nombres_archivos).

    Raises:
        FileNotFoundError: Si la carpeta de documentos no existe.
        ValueError: Si no hay PDFs en la carpeta.
        Exception: Si falla la conexión con la API de Gemini.
    """
    documentos = cargar_documentos(docs_path)
    chunks = crear_chunks(documentos)
    vectorstore = construir_vectorstore(chunks, api_key)
    cadena = construir_cadena_rag(vectorstore, api_key)

    nombres = list({os.path.basename(d.metadata.get("source", "")) for d in documentos})

    return cadena, len(documentos), nombres


def consultar(cadena: ConversationalRetrievalChain, pregunta: str) -> dict:
    """Realiza una consulta al sistema RAG.

    Args:
        cadena: Cadena RAG inicializada.
        pregunta: Pregunta del usuario en lenguaje natural.

    Returns:
        Diccionario con 'answer' y 'source_documents'.

    Raises:
        Exception: Si falla la consulta a la API de Gemini.
    """
    if not pregunta.strip():
        return {"answer": "Por favor, escribe una pregunta.", "source_documents": []}

    resultado = cadena({"question": pregunta})
    return resultado

