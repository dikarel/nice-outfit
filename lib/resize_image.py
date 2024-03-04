from PIL.Image import Image as PILImage
from math import sqrt


def resize_image(image: PILImage, ideal_number_of_pixels: int = 512**2) -> PILImage:
    image_number_of_pixels = image.width * image.height
    image_scaling_factor = sqrt(ideal_number_of_pixels / image_number_of_pixels)

    new_width = round(image.width * image_scaling_factor)
    new_height = round(image.height * image_scaling_factor)

    image = image.resize((new_width, new_height))
    return image
