from torch import cuda, float16


def cuda_if_available() -> str:
    return "cuda" if cuda.is_available() else "cpu"


def fp16_if_available() -> str | None:
    return "fp16" if cuda.is_available() else None


def float16_if_available():
    return float16 if cuda.is_available() else None
