from PIL.Image import Image as PILImage
from functools import cache
from diffusers import StableDiffusionInpaintPipeline


def redraw_image(prompt: str, image: PILImage, mask: PILImage) -> PILImage:
    inpaint_model = get_inpaint_model()

    return inpaint_model(
        prompt=prompt,
        image=image,
        mask_image=mask,
        width=image.width,
        height=image.height,
    ).images[0]


@cache
def get_inpaint_model():
    return StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting"
    ).to("cuda")
