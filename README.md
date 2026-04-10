# 📚 RAG Khan Academy

El objetivo de este proyecto es construir un sistema de Retrieval-Augmented Generation (RAG) que permita a los usuarios interactuar en lenguaje natural con el contenido educativo de los cursos de Khan Academy, a través de sus transcripciones en texto.

## Fuente de datos
- Dataset: iblai/ibl-khanacademy-transcripts
- Formato: JSON con campos como title, content, language, url, etc.

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado y en ejecución

## Instalación

```bash
# 1. Clona o descarga el proyecto
cd rag-khanacademy

# 2. Crea el entorno virtual
python -m venv venv
source venv/bin/activate   

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Descarga los modelos de Ollama
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Ejecución (en orden)

```bash
# Fase 1 — Limpieza del dataset
python clean_json.py        # → genera train_clean.json

# Fase 2 — Indexación
python index_data.py        # → genera ./chroma_db

# Fase 3+4 — Aplicación
python app.py               # → abre http://localhost:7860
```

## Parámetros configurables desde la interfaz

| Parámetro | Descripción |
|-----------|-------------|
| **k** | Número de fragmentos recuperados (1–10) |
| **Umbral de similitud** | Puntuación mínima para incluir un fragmento (0.0–1.0) |
| **Filtro por tema** | Restringe la búsqueda a un área temática concreta |

## Estructura del proyecto

```
rag-khanacademy/
├── clean_json.py      # Limpieza del dataset
├── index_data.py      # Generación de embeddings e indexación
├── app.py             # Pipeline RAG + interfaz Gradio
├── train_clean.json   # Dataset limpio (generado)
├── chroma_db/         # Base de datos vectorial (generada)
├── requirements.txt
└── README.md
```
