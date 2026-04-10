import gradio as gr

def responder(pregunta, historial):
    historial = historial or []
    historial = historial + [(pregunta, f"Respuesta a: {pregunta}")]
    return historial, ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chat", height=400)
    campo = gr.Textbox(placeholder="Escribe algo...", show_label=False)
    boton = gr.Button("Enviar")

    boton.click(
        fn=responder,
        inputs=[campo, chatbot],
        outputs=[chatbot, campo],
    )

demo.launch(server_name="127.0.0.1", server_port=7860, share=True)