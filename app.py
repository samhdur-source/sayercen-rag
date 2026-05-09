"""
app.py — SAYERCEN-D Technical Assistant
Chatbot RAG sobre documentos técnicos de SAYERCEN-D

Tecnologías: Streamlit · LangChain · Google Gemini · FAISS
Módulo: Python para IA — Proyecto Final Integrador
"""

import streamlit as st
import os
from rag_engine import inicializar_rag, consultar

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SAYERCEN-D Technical Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS personalizados ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Paleta SAYERCEN-D */
    :root {
        --gold: #C9A84C;
        --green-dark: #1A4731;
        --green-mid: #2E7D52;
        --green-light: #E8F5EE;
        --gray-light: #F7F8FA;
        --text-dark: #1A2B1F;
    }

    /* Header principal */
    .sayercen-header {
        background: linear-gradient(135deg, #1A4731 0%, #2E7D52 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #C9A84C;
    }
    .sayercen-header h1 {
        color: white;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 700;
    }
    .sayercen-header p {
        color: #A8D5B8;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }

    /* Tarjeta de fuentes */
    .source-card {
        background: #F0F9F4;
        border: 1px solid #B8DEC8;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: #2E7D52;
    }

    /* Badge de documentos */
    .doc-badge {
        background: #E8F5EE;
        border: 1px solid #2E7D52;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        color: #1A4731;
        display: inline-block;
        margin: 0.2rem;
    }

    /* Preguntas de demo */
    .demo-question {
        background: #FDF8EC;
        border-left: 3px solid #C9A84C;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        cursor: pointer;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── API Key: primero secrets, luego sidebar ─────────────────────────────────
def obtener_api_key() -> str | None:
    """Obtiene la API Key desde st.secrets o desde la entrada manual del usuario."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return st.session_state.get("api_key_manual", None)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/biogas.png", width=60)
    st.markdown("### ⚙️ Configuración")

    # API Key manual (si no está en secrets)
    try:
        st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key configurada")
    except Exception:
        api_key_input = st.text_input(
            "API Key de Gemini",
            type="password",
            placeholder="AIza...",
            help="Obtén tu API Key en aistudio.google.com",
        )
        if api_key_input:
            st.session_state["api_key_manual"] = api_key_input

    st.markdown("---")

    # Estado del sistema
    st.markdown("### 📊 Estado del sistema")

    if "rag_inicializado" in st.session_state and st.session_state.rag_inicializado:
        st.success(f"✅ RAG activo")
        st.markdown(f"**Documentos cargados:** {st.session_state.num_docs}")
        st.markdown("**Archivos indexados:**")
        for nombre in st.session_state.nombres_docs:
            if nombre:
                st.markdown(f'<div class="doc-badge">📄 {nombre}</div>', unsafe_allow_html=True)
    else:
        st.warning("⏳ Sistema no inicializado")

    st.markdown("---")

    # Botón para reinicializar
    if st.button("🔄 Recargar documentos", use_container_width=True):
        for key in ["rag_cadena", "rag_inicializado", "mensajes"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Limpiar chat
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.mensajes = []
        if "rag_cadena" in st.session_state:
            del st.session_state["rag_cadena"]
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#888; text-align:center;'>
    SAYERCEN-D Technical Assistant<br>
    Proyecto Final — Máster en IA<br>
    Powered by Gemini + LangChain
    </div>
    """, unsafe_allow_html=True)


# ── Header principal ────────────────────────────────────────────────────────
st.markdown("""
<div class="sayercen-header">
    <h1>🌱 SAYERCEN-D Technical Assistant</h1>
    <p>Asistente de consulta técnica sobre biodigestores, PTAR y energías renovables · 21 años de experiencia · 400+ proyectos</p>
</div>
""", unsafe_allow_html=True)


# ── Inicialización del RAG ──────────────────────────────────────────────────
api_key = obtener_api_key()

if not api_key:
    st.warning("⚠️ Configura tu API Key de Gemini en la barra lateral para continuar.")
    st.stop()

if "rag_inicializado" not in st.session_state or not st.session_state.rag_inicializado:
    with st.spinner("⚙️ Cargando y procesando documentos técnicos... (primera vez puede tardar ~30 segundos)"):
        try:
            cadena, num_docs, nombres = inicializar_rag(api_key=api_key, docs_path="docs")
            st.session_state.rag_cadena = cadena
            st.session_state.rag_inicializado = True
            st.session_state.num_docs = num_docs
            st.session_state.nombres_docs = nombres
            st.success(f"✅ {num_docs} páginas indexadas correctamente desde {len(nombres)} documentos.")
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.info("Crea una carpeta llamada 'docs' en el directorio del proyecto y añade tus PDFs.")
            st.stop()
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error al inicializar el RAG: {e}")
            st.info("Verifica que tu API Key de Gemini sea válida y tenga cuota disponible.")
            st.stop()


# ── Preguntas de demo ───────────────────────────────────────────────────────
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if not st.session_state.mensajes:
    st.markdown("#### 💡 Preguntas de demostración")
    col1, col2, col3 = st.columns(3)

    preguntas_demo = [
        "¿Qué es el Reactor Hyper Kinetik y cuál es su capacidad típica?",
        "¿Cómo funciona el sistema TA360° para agitación de biodigestores?",
        "¿Cuáles son los parámetros de diseño de la PTAR para el Rastro TIF Betulia?",
        "¿Qué establece la NOM-001-SEMARNAT-2021 para rastros?",
        "¿Cuál es la producción estimada de biogás del sistema instalado?",
        "¿Qué es la membrana BSTM y para qué se usa?",
    ]

    for i, (col, preg) in enumerate(zip([col1, col2, col3, col1, col2, col3], preguntas_demo)):
        with col:
            if st.button(f"💬 {preg[:55]}...", key=f"demo_{i}", use_container_width=True):
                st.session_state.pregunta_demo = preg
                st.rerun()


# ── Historial de chat ───────────────────────────────────────────────────────
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        if mensaje["role"] == "assistant" and "fuentes" in mensaje and mensaje["fuentes"]:
            with st.expander("📄 Fuentes consultadas", expanded=False):
                fuentes_vistas = set()
                for doc in mensaje["fuentes"]:
                    fuente = os.path.basename(doc.metadata.get("source", "Documento"))
                    pagina = doc.metadata.get("page", "?")
                    clave = f"{fuente}_p{pagina}"
                    if clave not in fuentes_vistas:
                        fuentes_vistas.add(clave)
                        st.markdown(
                            f'<div class="source-card">📄 <b>{fuente}</b> — Página {pagina}</div>',
                            unsafe_allow_html=True
                        )


# ── Input del usuario ───────────────────────────────────────────────────────
# Manejar pregunta demo si fue seleccionada
pregunta_inicial = st.session_state.pop("pregunta_demo", None)

prompt = st.chat_input("Escribe tu pregunta técnica aquí...") or pregunta_inicial

if prompt:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"role": "user", "content": prompt})

    # Generar respuesta RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Consultando documentos técnicos..."):
            try:
                resultado = consultar(st.session_state.rag_cadena, prompt)
                respuesta = resultado.get("answer", "No pude generar una respuesta.")
                fuentes = resultado.get("source_documents", [])

                st.markdown(respuesta)

                if fuentes:
                    with st.expander("📄 Fuentes consultadas", expanded=False):
                        fuentes_vistas = set()
                        for doc in fuentes:
                            fuente = os.path.basename(doc.metadata.get("source", "Documento"))
                            pagina = doc.metadata.get("page", "?")
                            clave = f"{fuente}_p{pagina}"
                            if clave not in fuentes_vistas:
                                fuentes_vistas.add(clave)
                                st.markdown(
                                    f'<div class="source-card">📄 <b>{fuente}</b> — Página {pagina}</div>',
                                    unsafe_allow_html=True
                                )

                st.session_state.mensajes.append({
                    "role": "assistant",
                    "content": respuesta,
                    "fuentes": fuentes,
                })

            except Exception as e:
                error_msg = f"❌ Error al consultar: {e}. Verifica tu API Key y conexión a internet."
                st.error(error_msg)
                st.session_state.mensajes.append({
                    "role": "assistant",
                    "content": error_msg,
                    "fuentes": [],
                })
