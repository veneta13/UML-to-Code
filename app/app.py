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
        image = gr.Image(type='numpy')
        output = gr.Textbox(
            label='Python Code',
            placeholder='class Example:\n'
                        '   def __init__(self, attribute):\n'
                        '       self.my_example_attribute = attribute\n'
                        '\n'
                        '   def set_example(self, example):\n'
                        '       self.example = example\n',
            show_copy_button=True
        )

    gr.Examples(
        examples=[
            os.path.join(os.path.dirname(__file__), "../diagrams/airline.jpg"),
            os.path.join(os.path.dirname(__file__), "../diagrams/two_classes_no_inher.jpg")
        ],
        inputs=image,
        outputs=None,
        fn=process_image
    )

    btn = gr.Button("Generate Python code")
    btn.click(fn=process_image, inputs=image, outputs=output)

app.launch()
