from PIL.Image import Image as PILImage
from functools import cache
from diffusers import StableDiffusionInpaintPipeline
from lib.optimize import optimize_sd_model
from lib.pad_image import pad_image, unpad_image
from lib.cuda import cuda_if_available, float16_if_available, fp16_if_available


def redraw_image(prompt: str, image: PILImage, mask: PILImage) -> PILImage:
    inpaint_model = get_inpaint_model()
    original_image_size = image.size

    # Stable Diffusion in-painting requires the image dimensions to be a multiple of 8
    image = pad_image(image, to_nearest_multiple_of=8)

    output = inpaint_model(
        prompt=prompt,
        image=image,
        mask_image=mask,
        width=image.width,
        height=image.height,
        num_inference_steps=30,
    ).images[0]

    return unpad_image(output, original_image_size)


@cache
def get_inpaint_model():
    inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision=fp16_if_available(),
        torch_dtype=float16_if_available(),
    ).to(cuda_if_available())

    return optimize_sd_model(inpaint_model)
