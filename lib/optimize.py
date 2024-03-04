from diffusers import StableDiffusionInpaintPipeline
from DeepCache import DeepCacheSDHelper
from importlib.util import find_spec


def optimize_sd_model(
    sd_model: StableDiffusionInpaintPipeline,
):
    """
    Optimizes a Stable Diffusion model for faster inference and less memory usage,
    and then returns the model
    """

    xformers_available = find_spec("xformers") is not None

    if xformers_available:
        sd_model.enable_xformers_memory_efficient_attention()

    helper = DeepCacheSDHelper(pipe=sd_model)
    helper.set_params(
        cache_interval=3,
        cache_branch_id=0,
    )

    return sd_model
