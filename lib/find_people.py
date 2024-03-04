from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL.Image import Image as PILImage
from PIL import Image
from lib.cloth_seg import everyhing_but_background_face_and_hair
from torch import zeros_like
from torch.nn.functional import interpolate
from functools import cache
from lib.cuda import cuda_if_available


def find_people(image: PILImage) -> PILImage:
    processor = get_seg_processor()
    model = get_seg_model()

    inputs = processor(images=image, return_tensors="pt").to(cuda_if_available())
    logits = model(**inputs).logits.cpu()

    upsampled_logits = interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    predictions = upsampled_logits.argmax(dim=1)[0]
    mask = zeros_like(predictions)

    for type in everyhing_but_background_face_and_hair():
        mask += (predictions == type.value).long()

    return Image.fromarray((mask * 255).byte().numpy(), "L")


@cache
def get_seg_processor():
    return SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes", device=cuda_if_available()
    )


@cache
def get_seg_model():
    return AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    ).to(cuda_if_available())
