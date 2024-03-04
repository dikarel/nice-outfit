# Nice outfit!

## Quickstart

1. `python -m venv venv`
2. `pip install -r requirements.txt`
3. `python app.py`

## How it works

1. A [fine-tuned SegFormer](https://huggingface.co/mattmdjaga/segformer_b2_clothes) is used to detect clothing in the image, generating a mask
2. This mask is then fed into a [Stable Diffusion in-paint pipeline](https://huggingface.co/runwayml/stable-diffusion-inpainting), in addition to the input image + outfit (as a text prompt), to generate the output
