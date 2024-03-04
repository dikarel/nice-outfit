from PIL import Image


def pad_image(image: Image.Image, to_nearest_multiple_of: int = 8) -> Image.Image:
    # No-op if the image size is already divisible by the target multiple
    if (
        image.width % to_nearest_multiple_of == 0
        and image.height % to_nearest_multiple_of == 0
    ):
        return image.copy()

    # Make a container that exactly bounds the image
    ctr_width = image.width + (
        to_nearest_multiple_of - image.width % to_nearest_multiple_of
    )
    ctr_height = image.height + (
        to_nearest_multiple_of - image.height % to_nearest_multiple_of
    )
    ctr = Image.new(image.mode, (ctr_width, ctr_height))

    # Paste the image exactly at the top lfet of the container
    ctr.paste(image, (0, 0))

    return ctr


def unpad_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    # No-op if the image size is already equal to the target size
    if image.size == size:
        return image.copy()

    return image.crop((0, 0, size[0], size[1]))
