from PIL import Image, ImageFilter


def expand_mask(mask: Image.Image, factor: int = 5):
    mask = mask.filter(ImageFilter.MaxFilter(factor * 2 + 1))
    mask = mask.filter(ImageFilter.GaussianBlur(factor / 2))
    return mask
