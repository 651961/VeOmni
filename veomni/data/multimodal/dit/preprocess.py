# dit preprocess should not be used for llm or mllms
import os

from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("Tom-and-Jerry-VideoGeneration-Dataset")
def tom_and_jerry_preprocess(conversations, **kwargs):
    prompt = conversations["prompt"]
    outputs = {}
    images = {}
    videos = [conversations["video_bytes"]]
    return prompt, outputs, images, videos


@PREPROCESSOR_REGISTRY.register("Qwen-Image")
@PREPROCESSOR_REGISTRY.register("QwenImage")
def qwen_image_preprocess(conversations, **kwargs):
    prompt = next(
        (conversations[key] for key in ("prompt", "text", "caption") if conversations.get(key) is not None), None
    )
    image = next(
        (
            conversations[key]
            for key in ("image", "image_bytes", "image_path", "target_image")
            if conversations.get(key) is not None
        ),
        None,
    )
    if prompt is None:
        raise ValueError("Qwen-Image data requires one of: prompt, text, caption.")
    if image is None:
        raise ValueError("Qwen-Image data requires one of: image, image_bytes, image_path, target_image.")
    if not isinstance(prompt, str) or not isinstance(image, (str, bytes)):
        raise ValueError("Qwen-Image requires a text prompt and an image path, URL, or bytes.")
    data_dir = kwargs.get("data_dir", "")
    if data_dir and isinstance(image, str) and not os.path.isabs(image) and "://" not in image:
        image = os.path.join(data_dir, image)
    return [("text", prompt)], [("image", None)], [image], []


@PREPROCESSOR_REGISTRY.register("Qwen-Image-Edit")
@PREPROCESSOR_REGISTRY.register("QwenImageEdit")
def qwen_image_edit_preprocess(conversations, **kwargs):
    prompt = next(
        (
            conversations[key]
            for key in ("prompt", "text", "caption", "instruction")
            if conversations.get(key) is not None
        ),
        None,
    )
    target_image = next(
        (
            conversations[key]
            for key in ("target_image", "image", "image_bytes", "image_path")
            if conversations.get(key) is not None
        ),
        None,
    )
    edit_images = None
    for key in (
        "edit_images",
        "source_images",
        "src_images",
        "edit_image",
        "source_image",
        "src_image",
    ):
        value = conversations.get(key)
        if value is not None:
            edit_images = value if isinstance(value, list) else [value]
            break
    if prompt is None:
        raise ValueError("Qwen-Image-Edit data requires one of: prompt, text, caption, instruction.")
    if target_image is None:
        raise ValueError("Qwen-Image-Edit data requires one of: target_image, image, image_bytes, image_path.")
    if not edit_images:
        raise ValueError("Qwen-Image-Edit data requires one of: edit_image(s), source_image(s), src_image(s).")
    if not isinstance(prompt, str):
        raise ValueError("Qwen-Image-Edit requires a text prompt.")
    if not isinstance(target_image, (str, bytes)):
        raise ValueError("Qwen-Image-Edit target must be an image path, URL, or bytes.")
    if not all(isinstance(img, (str, bytes)) for img in edit_images):
        raise ValueError("Qwen-Image-Edit source images must be paths, URLs, or bytes.")

    data_dir = kwargs.get("data_dir", "")

    def _resolve(path):
        if data_dir and isinstance(path, str) and not os.path.isabs(path) and "://" not in path:
            return os.path.join(data_dir, path)
        return path

    target_image = _resolve(target_image)
    edit_images = [_resolve(img) for img in edit_images]
    inputs = [("image", None)] * len(edit_images) + [("text", prompt)]
    return inputs, [("image", None)], edit_images + [target_image], []


@PREPROCESSOR_REGISTRY.register("minimax_h3")
def minimax_h3_preprocess(conversations, **kwargs):
    data_dir = kwargs.get("data_dir", "")
    prompt = conversations["prompt"]

    # Video path
    video_path = conversations.get("video", "")
    if video_path and data_dir:
        video_path = os.path.join(data_dir, video_path)

    # Audio path (optional)
    audio_path = conversations.get("input_audio", "")
    if audio_path and data_dir:
        audio_path = os.path.join(data_dir, audio_path)

    audios = {"audio": audio_path} if audio_path else {}
    videos = [video_path] if video_path else []

    # FL2VA keyframes extracted from video frames in condition_model.get_condition()
    return prompt, audios, [], videos


@PREPROCESSOR_REGISTRY.register("minimax_h3_refinement")
def minimax_h3_refinement_preprocess(conversations, **kwargs):
    """Read paired low-quality/high-quality videos for H3 refinement training.

    ``video`` is the clean target and ``input_video`` is the low-quality source
    that is RGB-upsampled before its VAE encoding.  Keeping the two paths in
    the raw example lets the refinement condition model reproduce LongCat's
    ``x_up = VAE(Upsample(Decode(x_lr)))`` path without putting the target into
    the conditioning stream.
    """
    data_dir = kwargs.get("data_dir", "")
    prompt = conversations.get("prompt", conversations.get("text", ""))

    def resolve(path):
        if path and data_dir and not os.path.isabs(path) and "://" not in path:
            return os.path.join(data_dir, path)
        return path

    target_path = resolve(conversations.get("video", conversations.get("target_video", "")))
    input_path = resolve(conversations.get("input_video", conversations.get("input", "")))
    if not target_path or not input_path:
        raise ValueError("minimax_h3_refinement requires both `video` and `input_video` paths.")

    audio_path = resolve(conversations.get("input_audio", ""))
    audios = {"audio": audio_path} if audio_path else {}
    return prompt, audios, [], [target_path], [input_path]
