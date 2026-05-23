import gradio as gr
from n2_backend import chat

CSS = """
footer {
    display: none;
}

.gradio-container {
    max-width: 1400px !important;
}

.message.user {
    background: #2b313e !important;
    border-radius: 12px !important;
}

.message.bot {
    background: #1f2430 !important;
    border-radius: 12px !important;
}

.sidebar {
    border-right: 1px solid #333;
    padding-right: 10px;
}
"""

def clear_chat():
    return [], [], [], None

with gr.Blocks(
    title="Qwen3-VL Assistant"
) as demo:
    gr.Markdown("""
        # 🤖 Qwen3-VL Vision Assistant
    
        Chat multimodal con imágenes usando Qwen3-VL + LM Studio
    """)

    with gr.Row():

        # SIDEBAR
        with gr.Column(scale=1, min_width=250, elem_classes="sidebar"):

            gr.Markdown("## ⚙️ Configuración")

            temperature = gr.Slider(
                0,
                2,
                value=0.1,
                step=0.1,
                label="Temperature"
            )

            max_tokens = gr.Slider(
                64,
                4096,
                value=128,
                step=64,
                label="Max Tokens"
            )

            clear_btn = gr.Button(
                "🗑️ Limpiar chat",
                variant="secondary"
            )

        # CHAT
        with gr.Column(scale=4):

            chatbot = gr.Chatbot(
                height=700,
                avatar_images=(
                    None,
                    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
                ),
                render_markdown=True
            )

            chat_input = gr.MultimodalTextbox(
                file_types=["image"],
                placeholder="Escribe un mensaje o arrastra imágenes...",
                show_label=False,
                lines=3
            )

            chat_interface = gr.ChatInterface(
                fn=chat,
                chatbot=chatbot,
                textbox=chat_input,
                multimodal=True,
                additional_inputs=[
                    temperature,
                    max_tokens
                ],
                fill_height=True,
                examples=[
                    ["Dime el nombre de esta planta.", 0, 1024],
                    ["¿Qué planta es esta?", 0, 1024],
                    ["Identifica esta planta y dame información de ella.", 0, 1024],
                    ["¿Qué datos tienes de la planta anterior?", 0, 1024]
                ]
            )

    clear_btn.click(
        clear_chat,
        outputs=[
            chatbot,
            chat_interface.chatbot_state,
            chat_interface.chatbot_value,
            chat_interface.saved_input
        ]
    )

demo.launch(
    theme=gr.themes.Monochrome(),
    css=CSS
)
