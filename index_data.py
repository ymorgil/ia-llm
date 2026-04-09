"""
index_data.py
=============
Fase 2: Genera embeddings de cada fragmento de transcripción
y los indexa en una base de datos vectorial ChromaDB (persistente en disco).

Requisito previo:
    ollama pull nomic-embed-text
"""

import json
import os
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ── Configuración ─────────────────────────────────────────────────────────────
INPUT_FILE       = "train_clean.json"
CHROMA_DIR       = "./chroma_db"          # directorio de persistencia
COLLECTION_NAME  = "khanacademy"
EMBED_MODEL      = "nomic-embed-text"     # modelo de embeddings en Ollama
BATCH_SIZE       = 50                     # documentos por lote (evita sobrecarga)
# ──────────────────────────────────────────────────────────────────────────────


def cargar_documentos(ruta: str) -> list[Document]:
    """
    Carga el JSON limpio y convierte cada registro en un objeto Document
    de LangChain, con metadatos para poder filtrar después.
    """
    with open(ruta, "r", encoding="utf-8") as f:
        registros = json.load(f)

    documentos = []
    for r in registros:
        doc = Document(
            page_content=r["content"],
            metadata={
                "title":    r.get("title", ""),
                "url":      r.get("url", ""),
                "language": r.get("language", ""),
                # Extraer el "topic" a partir del título (primeras dos palabras)
                # Puedes ajustar esta lógica según los datos reales
                "topic":    " ".join(r.get("title", "").split()[:3]),
            },
        )
        documentos.append(doc)

    return documentos


def indexar_en_lotes(vectorstore: Chroma, documentos: list[Document], batch_size: int):
    """
    Añade documentos a ChromaDB en lotes para no saturar la memoria
    ni sobrecargar el servidor de embeddings.
    """
    total = len(documentos)
    for inicio in tqdm(range(0, total, batch_size), desc="Indexando lotes"):
        lote = documentos[inicio : inicio + batch_size]
        vectorstore.add_documents(lote)


def main():
    print("=" * 60)
    print("  Fase 2 — Indexación en ChromaDB")
    print("=" * 60)

    # ── 1. Verificar que existe el JSON limpio ────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ No se encontró {INPUT_FILE}.")
        print("   Ejecuta primero: python clean_json.py")
        return

    # ── 2. Cargar documentos ──────────────────────────────────────────────────
    print(f"\n📂 Cargando documentos desde {INPUT_FILE}...")
    documentos = cargar_documentos(INPUT_FILE)
    print(f"   Documentos cargados: {len(documentos)}")

    # ── 3. Inicializar modelo de embeddings ───────────────────────────────────
    print(f"\n🔗 Conectando con Ollama (modelo: {EMBED_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Test rápido para verificar conexión con Ollama
    try:
        _ = embeddings.embed_query("test de conexión")
        print("   ✅ Conexión con Ollama correcta")
    except Exception as e:
        print(f"\n❌ Error conectando con Ollama: {e}")
        print("   Asegúrate de que Ollama está en ejecución: ollama serve")
        return

    # ── 4. Crear o cargar la colección ChromaDB ───────────────────────────────
    print(f"\n💾 Iniciando ChromaDB en: {CHROMA_DIR}")

    # Si ya existe la BD, la borramos para reindexar desde cero
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("   ⚠️  Base de datos anterior eliminada (reindexando)")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # ── 5. Indexar documentos ─────────────────────────────────────────────────
    print(f"\n⚙️  Generando embeddings e indexando ({len(documentos)} documentos)...")
    print("   Esto puede tardar varios minutos según el hardware.")
    indexar_en_lotes(vectorstore, documentos, BATCH_SIZE)

    # ── 6. Verificación final ─────────────────────────────────────────────────
    total_indexados = vectorstore._collection.count()
    print(f"\n✅ Indexación completada:")
    print(f"   Documentos en la BD : {total_indexados}")
    print(f"   Directorio ChromaDB : {CHROMA_DIR}")

    # Prueba de recuperación
    print("\n🔍 Prueba de recuperación semántica:")
    resultados = vectorstore.similarity_search("what is a fraction?", k=2)
    for i, doc in enumerate(resultados, 1):
        print(f"\n   [{i}] {doc.metadata.get('title', 'Sin título')}")
        print(f"       {doc.page_content[:120]}...")


if __name__ == "__main__":
    main()
