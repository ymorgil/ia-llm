# Aplicación principal (app.py): pipeline RAG + interfaz Gradio completamente funcional
"""
Requisitos previos:
    1. python clean_json.py      → genera train_clean.json
    2. python index_data.py      → genera ./chroma_db
    3. ollama pull llama3.2
    4. ollama pull nomic-embed-text
"""

import json
import gradio as gr
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "khanacademy"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "llama3.2"

# Prompt del sistema 
PROMPT_TEMPLATE = """Eres un asistente educativo especializado en el contenido de Khan Academy.
Responde ÚNICAMENTE basándote en el contexto proporcionado a continuación.
Si la respuesta no se encuentra en el contexto, indica claramente:
"No dispongo de información sobre ese tema en el material disponible."

Contexto recuperado:
{context}

Historial de conversación:
{chat_history}

Pregunta: {question}

Respuesta (basada exclusivamente en el contexto):"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_TEMPLATE,
)

# Carga los temas disponibles desde el JSON limpio para el dropdown.
def cargar_temas() -> list[str]:
    try:
        with open("train_clean.json", "r", encoding="utf-8") as f:
            datos = json.load(f)
        # Extraer primeras 3 palabras del título como "tema"
        temas = sorted(set(
            " ".join(r.get("title", "").split()[:3])
            for r in datos
            if r.get("title")
        ))
        return ["Todos"] + temas
    except FileNotFoundError:
        return ["Todos"]

# Crea y devuelve una ConversationalRetrievalChain con los parámetros dados, cada vez que el usuario cambia parámetros del retriever
def crear_chain(k: int, score_threshold: float, llm_model: str):
    # Embeddings y vectorstore
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Retriever con parámetros configurables
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
        },
    )

    # LLM local vía Ollama
    llm = OllamaLLM(model=llm_model, temperature=0.1)

    # Memoria de conversación (persiste entre turnos)
    memoria = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Chain conversacional RAG
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memoria,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False,
    )

    return chain

# Formatea los documentos recuperados para mostrarlos en la interfaz.

def formatear_fuentes(documentos: list) -> str:
    if not documentos:
        return "⚠️ No se recuperaron fragmentos relevantes para esta consulta."

    texto = ""
    for i, doc in enumerate(documentos, 1):
        titulo = doc.metadata.get("title", "Sin título")
        url    = doc.metadata.get("url", "")
        extracto = doc.page_content[:300].strip()

        texto += f"**[{i}] {titulo}**\n"
        if url:
            texto += f"🔗 {url}\n"
        texto += f"_{extracto}..._\n\n"

    return texto.strip()

estado = {
    "chain":   None,
    "k":       3,
    "threshold": 0.3,
    "topic":   "Todos",
}

# Inicializa o reinicializa la chain con los parámetros actuales.
def inicializar_chain(k, threshold):
    try:
        estado["chain"]     = crear_chain(k, threshold, LLM_MODEL)
        estado["k"]         = k
        estado["threshold"] = threshold
        return True
    except Exception as e:
        print(f"Error al crear la chain: {e}")
        return False


def responder(
    pregunta: str,
    historial: list,
    k: int,
    threshold: float,
    topic: str,
):
    if not pregunta.strip():
        return historial, historial, "Escribe una pregunta primero."

    # Reinicializar la chain si cambiaron los parámetros
    if (
        estado["chain"] is None
        or estado["k"] != k
        or estado["threshold"] != threshold
    ):
        ok = inicializar_chain(k, threshold)
        if not ok:
            msg = "❌ Error al conectar con Ollama. Comprueba que está en ejecución."
            historial.append({"role": "user", "content": pregunta})
            historial.append({"role": "assistant", "content": respuesta})
            return historial, historial, fuentes_txt

    # Añadir filtro por topic si no es "Todos"
    if topic != "Todos":
        estado["chain"].retriever.search_kwargs["filter"] = {"topic": topic}
    else:
        estado["chain"].retriever.search_kwargs.pop("filter", None)

    # Invocar la chain
    try:
        resultado   = estado["chain"].invoke({"question": pregunta})
        respuesta   = resultado.get("answer", "No se obtuvo respuesta.")
        documentos  = resultado.get("source_documents", [])
        fuentes_txt = formatear_fuentes(documentos)
    except Exception as e:
        respuesta   = f"❌ Error al procesar la pregunta: {e}"
        fuentes_txt = ""

    # Actualizar historial del chatbot
    historial.append((pregunta, respuesta))
    return historial, historial, fuentes_txt

# Resetea la memoria y el historial visible.
def limpiar_historial():
    if estado["chain"] is not None:
        estado["chain"].memory.clear()
    estado["chain"] = None  # forzar reinicialización
    return [], [], ""


# Interfaz Gradio
def construir_interfaz():
    temas = cargar_temas()

    with gr.Blocks(title="RAG Khan Academy") as demo:

        # Encabezado 
        gr.Markdown("""
        # 📚 Asistente RAG — Khan Academy
        Haz preguntas sobre el contenido educativo de Khan Academy.
        Las respuestas se generan **exclusivamente** a partir de las transcripciones indexadas.
        """)

        # Layout principal 
        with gr.Row():

            # Panel izquierdo: chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversación",
                    height=480,
                    show_label=True,
                )   

                with gr.Row():
                    campo_pregunta = gr.Textbox(
                        placeholder="Escribe tu pregunta aquí...",
                        show_label=False,
                        scale=4,
                        lines=1,
                    )
                    btn_enviar = gr.Button("Enviar ▶", variant="primary", scale=1)

                btn_limpiar = gr.Button("🗑️ Limpiar historial", variant="secondary")

            # Panel derecho: fuentes y parámetros
            with gr.Column(scale=2):

                gr.Markdown("### ⚙️ Parámetros del retriever")

                slider_k = gr.Slider(
                    minimum=1, maximum=10, step=1, value=3,
                    label="Fragmentos a recuperar (k)",
                    info="Mayor k = más contexto, pero más lento",
                )
                slider_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                    label="Umbral de similitud mínima",
                    info="Mayor umbral = fragmentos más relevantes, puede no devolver nada",
                )
                dropdown_topic = gr.Dropdown(
                    choices=temas,
                    value="Todos",
                    label="Filtrar por tema",
                    info="Filtra los fragmentos según el área temática",
                )

                gr.Markdown("### 📄 Fuentes recuperadas")
                panel_fuentes = gr.Markdown(
                    value="Las fuentes aparecerán aquí después de tu primera pregunta.",
                    elem_classes=["fuentes-panel"],
                )

        # Estado interno (historial LangChain) 
        state_historial = gr.State([])
        # Eventos 
        def on_enviar(pregunta, historial, k, threshold, topic):
            if not isinstance(historial, list):
                historial = []
            return responder(pregunta, historial, k, threshold, topic)

        btn_enviar.click(
            fn=on_enviar,
            inputs=[campo_pregunta, state_historial, slider_k, slider_threshold, dropdown_topic],
            outputs=[chatbot, state_historial, panel_fuentes],
        ).then(
            fn=lambda: "",
            outputs=campo_pregunta,   # limpiar campo de texto tras enviar
        )

        # Enviar también con Enter
        campo_pregunta.submit(
            fn=on_enviar,
            inputs=[campo_pregunta, state_historial, slider_k, slider_threshold, dropdown_topic],
            outputs=[chatbot, state_historial, panel_fuentes],
        ).then(
            fn=lambda: "",
            outputs=campo_pregunta,
        )

        btn_limpiar.click(
            fn=limpiar_historial,
            outputs=[chatbot, state_historial, panel_fuentes],
        )

    return demo


# Punto de entrada 
if __name__ == "__main__":
    import os

    # Verificar que la base de datos existe
    if not os.path.exists(CHROMA_DIR):
        print("❌ No se encontró la base de datos vectorial.")
        print("   Ejecuta primero:")
        print("     python clean_json.py")
        print("     python index_data.py")
        exit(1)

    print("🚀 Iniciando aplicación RAG Khan Academy...")
    print(f"   LLM        : {LLM_MODEL}")
    print(f"   Embeddings : {EMBED_MODEL}")
    print(f"   ChromaDB   : {CHROMA_DIR}")

    demo = construir_interfaz()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
    )