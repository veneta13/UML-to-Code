import os

import gradio as gr
from app_helper import get_class_text, build_prompt, generate_code


def process_image(image):
    class_text = get_class_text(image)
    prompt = build_prompt(class_text)
    return generate_code(prompt)


with gr.Blocks() as app:
    gr.Markdown('# UML-to-Python Generator')

    with gr.Row():
        image = gr.Image(type='numpy', height=600)
        output = gr.Textbox()

    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "../diagrams/airline.jpg")],
        inputs=image,
        outputs=None,
        fn=process_image
    )

    btn = gr.Button("Run")
    btn.click(fn=process_image, inputs=image, outputs=output)

app.launch()
