# dit preprocess should not be used for llm or mllms
from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("Tom-and-Jerry-VideoGeneration-Dataset")
def tom_and_jerry_preprocess(conversations, **kwargs):
    prompt = conversations["prompt"]
    outputs = {}
    images = {}
    videos = [conversations["video_bytes"]]
    return prompt, outputs, images, videos


@PREPROCESSOR_REGISTRY.register("Qwen-Image-Edit-2511")
@PREPROCESSOR_REGISTRY.register("Krea2-Image-Edit")
def qwen_image_edit_preprocess(conversations, **kwargs):
    """Preprocessor for image-edit raw parquet rows.

    Each row carries:
        - ``prompt``           : str
        - ``image_bytes``      : bytes (target image)
        - ``edit_image_bytes`` : list[bytes] (one or more reference images)

    Returns ``images`` as a role-keyed dict so the downstream ``dit_online``
    transform can fetch ``target`` and ``edit`` images separately while
    preserving order. The SFT recipe expects exactly one entry under
    ``image`` (the target) and one or more under ``edit_image`` (references).
    """
    prompt = conversations["prompt"]
    outputs = {}
    edit_bytes = conversations.get("edit_image_bytes") or []
    if isinstance(edit_bytes, (bytes, bytearray)):
        edit_bytes = [edit_bytes]
    images = {
        "image": [conversations["image_bytes"]],
        "edit_image": list(edit_bytes),
    }
    videos = []
    return prompt, outputs, images, videos


@PREPROCESSOR_REGISTRY.register("Qwen-Image")
@PREPROCESSOR_REGISTRY.register("QwenImage")
def qwen_image_preprocess(conversations, **kwargs):
    prompt = conversations.get("prompt") or conversations.get("text") or conversations.get("caption")
    image = (
        conversations.get("image")
        or conversations.get("image_bytes")
        or conversations.get("image_path")
        or conversations.get("target_image")
    )
    if prompt is None:
        raise ValueError("Qwen-Image data requires one of: prompt, text, caption.")
    if image is None:
        raise ValueError("Qwen-Image data requires one of: image, image_bytes, image_path, target_image.")
    return prompt, {}, [image], []
