# 🌱 SAYERCEN-D Technical Assistant

**Asistente de consulta técnica con IA** sobre biodigestores, plantas de tratamiento de aguas residuales (PTAR) y energías renovables, desarrollado con tecnología RAG (Retrieval-Augmented Generation).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tu-app.streamlit.app)

---

## ¿Qué hace esta aplicación?

SAYERCEN-D Technical Assistant es un chatbot conversacional que responde preguntas técnicas basándose **únicamente en documentos internos de SAYERCEN-D** — información que los modelos de IA generales no conocen. Combina:

- **Google Gemini 2.0 Flash** como modelo de lenguaje
- **LangChain** para orquestar el pipeline RAG
- **FAISS** como base vectorial local (sin servidor)
- **Streamlit** como interfaz web profesional

El sistema indexa automáticamente los PDFs técnicos de la carpeta `docs/`, crea embeddings semánticos y recupera los fragmentos más relevantes para cada pregunta antes de generar la respuesta.

---

## Tecnologías utilizadas

| Componente | Tecnología |
|-----------|-----------|
| Interfaz web | Streamlit |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google Embedding-001 |
| Vector store | FAISS (local) |
| Orquestación RAG | LangChain |
| Carga de PDFs | PyPDF |

---

## Estructura del proyecto

```
sayercen-rag/
├── app.py              # Interfaz Streamlit principal
├── rag_engine.py       # Motor RAG: carga, indexado y consulta
├── requirements.txt    # Dependencias Python
├── .streamlit/
│   └── secrets.toml    # API Key local (NO subir a GitHub)
├── docs/               # Carpeta con los PDFs técnicos
│   └── *.pdf
└── README.md
```

---

## Instalación y uso local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/sayercen-rag.git
cd sayercen-rag
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar API Key

Crea el archivo `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "tu-api-key-de-gemini"
```

> Obtén tu API Key gratis en [aistudio.google.com](https://aistudio.google.com/apikey)

### 4. Añadir documentos

Coloca tus archivos PDF en la carpeta `docs/`:

```
docs/
├── reactor-hyper-kinetik.pdf
├── sistema-ta360.pdf
├── propuesta-ptar-betulia.pdf
└── ... (mínimo 10 documentos)
```

### 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

---

## Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub (sin incluir `secrets.toml` ni la carpeta `docs/` si son confidenciales)
2. Ve a [share.streamlit.io](https://share.streamlit.io) y conecta tu repo
3. En **Settings → Secrets**, añade:
   ```toml
   GEMINI_API_KEY = "tu-api-key-aqui"
   ```
4. Sube la carpeta `docs/` con tus PDFs al repositorio (si no son confidenciales) o usa el uploader integrado

---

## Preguntas de demostración

El sistema incluye preguntas de demo preconfiguradas:

- ¿Qué es el Reactor Hyper Kinetik y cuál es su capacidad típica?
- ¿Cómo funciona el sistema TA360° para agitación de biodigestores?
- ¿Cuáles son los parámetros de diseño de la PTAR para el Rastro TIF Betulia?
- ¿Qué establece la NOM-001-SEMARNAT-2021 para rastros?
- ¿Cuál es la producción estimada de biogás del sistema instalado?
- ¿Qué es la membrana BSTM y para qué se usa?

---

## Acerca de SAYERCEN-D

**SAYERCEN-D** (Servicios Ambientales y de Energías Renovables del Centro) es una empresa mexicana con sede en León, Guanajuato, especializada en:

- Biodigestores tipo laguna y Reactor Hyper Kinetik
- Plantas de tratamiento de aguas residuales (PTAR)
- Sistemas de agitación TA360° para lagoons
- Membranas BSTM para captura de biogás
- Cumplimiento normativo NOM-001-SEMARNAT-2021

**21 años de experiencia | 400+ proyectos | Clientes: Bachoco, Nestlé, Walmart México**

---

## Proyecto académico

Desarrollado como **Proyecto Final Integrador** del módulo *Python para IA* del Máster en Inteligencia Artificial.

- **Tipo:** Sistema RAG (Retrieval-Augmented Generation)
- **Interfaz:** Streamlit
- **Deployment:** Streamlit Cloud
- **Autor:** Samuel H. Durán Rangel — Director SAYERCEN-D
