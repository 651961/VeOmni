from ...data_transform import DATA_TRANSFORM_REGISTRY
from ..image_utils import fetch_images
from ..preprocess import conv_preprocess
from ..video_utils import fetch_videos


@DATA_TRANSFORM_REGISTRY.register("dit_online")
def process_dit_online_example(example, source_name, **kwargs):
    inputs, outputs, images, videos = conv_preprocess(source=source_name, conversations=example, **kwargs)
    if kwargs.get("use_audio_in_video", False):
        raise NotImplementedError("Audio in video is not supported yet for dit training.")
    videos, _ = fetch_videos(videos, use_audio_in_video=False, **kwargs)
    # Some sources (e.g. Qwen-Image-Edit-2511) return a role-keyed dict
    # ``{"image": [...], "edit_image": [...]}`` so the downstream condition
    # model can distinguish the target image from reference images.
    if isinstance(images, dict):
        images = {role: fetch_images(list(v), **kwargs) for role, v in images.items()}
    else:
        images = fetch_images(images, **kwargs)
    processed_example = {
        "inputs": inputs,
        "outputs": outputs,
        "images": images,
        "videos": videos,
    }
    return [processed_example]


@DATA_TRANSFORM_REGISTRY.register("dit_offline")
def process_dit_offline_example(example, **kwargs):
    import pickle as pk

    processed_example = {key: pk.loads(value) for key, value in example.items()}
    return [processed_example]
