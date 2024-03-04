import gradio as gr
from random import choice
from lib.redraw_image import redraw_image
from lib.find_people import find_people
from PIL import Image
from PIL.Image import Image as PILImage
from lib.resize_image import resize_image

OUTFIT_SELECTION = [
    "Athletic wear",
    "Ball gown",
    "Bridal gown",
    "Camel trench coat",
    "Formal suit",
    "Haute couture",
    "Summer dress",
    "Winter coat",
]


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Image of yourself")

            with gr.Column():
                btn_change = gr.Button(value="Change outfit", variant="primary")
                drp_outfit = gr.Dropdown(
                    label="Select outfit",
                    choices=sorted(OUTFIT_SELECTION),
                    value=choice(OUTFIT_SELECTION),
                )
                img_output = gr.Image(label="Image of you wearing the outfit")

        btn_change.click(
            generate_output, inputs=[img_input, drp_outfit], outputs=[img_output]
        )

    demo.queue().launch()


def generate_output(img_input: PILImage, drp_outfit: str) -> PILImage:
    img_input = Image.fromarray(img_input)
    img_input = resize_image(img_input)

    people_mask = find_people(img_input)
    img_output = redraw_image(
        prompt=f"person wearing {drp_outfit}",
        image=img_input,
        mask=people_mask,
    )

    return img_output


if __name__ == "__main__":
    main()
