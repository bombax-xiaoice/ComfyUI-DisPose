# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copied from diffusers==0.27.0 in controlnet.py and single_file_utils.py as they no longer exists in latest diffusers versions

from huggingface_hub.utils import validate_hf_hub_args

from diffusers.loaders.single_file_utils import (
    logger,
    fetch_original_config,
    _extract_repo_id_and_weights_name,
    set_image_size,
    convert_controlnet_checkpoint,
)
import re
import os.path
from diffusers.models.modeling_utils import load_state_dict
from diffusers.utils.hub_utils import _get_model_file
from contextlib import nullcontext
from diffusers.utils.accelerate_utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights

def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if (
        "unet_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["unet_config"] is not None
    ):
        unet_params = original_config["model"]["params"]["unet_config"]["params"]
    else:
        unet_params = original_config["model"]["params"]["network_config"]["params"]

    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    if unet_params["transformer_depth"] is not None:
        transformer_layers_per_block = (
            unet_params["transformer_depth"]
            if isinstance(unet_params["transformer_depth"], int)
            else list(unet_params["transformer_depth"])
        )
    else:
        transformer_layers_per_block = 1

    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    head_dim = unet_params["num_heads"] if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params["use_linear_in_transformer"] if "use_linear_in_transformer" in unet_params else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim_mult = unet_params["model_channels"] // unet_params["num_head_channels"]
            head_dim = [head_dim_mult * c for c in list(unet_params["channel_mult"])]

    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None

    if unet_params["context_dim"] is not None:
        context_dim = (
            unet_params["context_dim"]
            if isinstance(unet_params["context_dim"], int)
            else unet_params["context_dim"][0]
        )

    if "num_classes" in unet_params:
        if unet_params["num_classes"] == "sequential":
            if context_dim in [2048, 1280]:
                # SDXL
                addition_embed_type = "text_time"
                addition_time_embed_dim = 256
            else:
                class_embed_type = "projection"
            assert "adm_in_channels" in unet_params
            projection_class_embeddings_input_dim = unet_params["adm_in_channels"]

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params["in_channels"],
        "down_block_types": down_block_types,
        "block_out_channels": block_out_channels,
        "layers_per_block": unet_params["num_res_blocks"],
        "cross_attention_dim": context_dim,
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "addition_embed_type": addition_embed_type,
        "addition_time_embed_dim": addition_time_embed_dim,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "transformer_layers_per_block": transformer_layers_per_block,
    }

    if "disable_self_attentions" in unet_params:
        config["only_cross_attention"] = unet_params["disable_self_attentions"]

    if "num_classes" in unet_params and isinstance(unet_params["num_classes"], int):
        config["num_class_embeds"] = unet_params["num_classes"]

    config["out_channels"] = unet_params["out_channels"]
    config["up_block_types"] = up_block_types

    return config

def create_controlnet_diffusers_config(original_config, image_size: int):
    unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    diffusers_unet_config = create_unet_diffusers_config(original_config, image_size=image_size)

    controlnet_config = {
        "conditioning_channels": unet_params["hint_channels"],
        "in_channels": diffusers_unet_config["in_channels"],
        "down_block_types": diffusers_unet_config["down_block_types"],
        "block_out_channels": diffusers_unet_config["block_out_channels"],
        "layers_per_block": diffusers_unet_config["layers_per_block"],
        "cross_attention_dim": diffusers_unet_config["cross_attention_dim"],
        "attention_head_dim": diffusers_unet_config["attention_head_dim"],
        "use_linear_projection": diffusers_unet_config["use_linear_projection"],
        "class_embed_type": diffusers_unet_config["class_embed_type"],
        "addition_embed_type": diffusers_unet_config["addition_embed_type"],
        "addition_time_embed_dim": diffusers_unet_config["addition_time_embed_dim"],
        "projection_class_embeddings_input_dim": diffusers_unet_config["projection_class_embeddings_input_dim"],
        "transformer_layers_per_block": diffusers_unet_config["transformer_layers_per_block"],
    }

    return controlnet_config

def load_single_file_model_checkpoint(
    pretrained_model_link_or_path,
    resume_download=False,
    force_download=False,
    proxies=None,
    token=None,
    cache_dir=None,
    local_files_only=None,
    revision=None,
):
    if os.path.isfile(pretrained_model_link_or_path):
        checkpoint = load_state_dict(pretrained_model_link_or_path)
    else:
        repo_id, weights_name = _extract_repo_id_and_weights_name(pretrained_model_link_or_path)
        checkpoint_path = _get_model_file(
            repo_id,
            weights_name=weights_name,
            force_download=force_download,
            cache_dir=cache_dir,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        checkpoint = load_state_dict(checkpoint_path)

    # some checkpoints contain the model state dict under a "state_dict" key
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    return checkpoint

def create_diffusers_controlnet_model_from_ldm(
    pipeline_class_name, original_config, checkpoint, upcast_attention=False, image_size=None, torch_dtype=None
):
    # import here to avoid circular imports
    from diffusers.models import ControlNetModel

    image_size = set_image_size(pipeline_class_name, original_config, checkpoint, image_size=image_size)

    diffusers_config = create_controlnet_diffusers_config(original_config, image_size=image_size)
    diffusers_config["upcast_attention"] = upcast_attention

    diffusers_format_controlnet_checkpoint = convert_controlnet_checkpoint(checkpoint, diffusers_config)

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        controlnet = ControlNetModel(**diffusers_config)

    if is_accelerate_available():
        from diffusers.models.modeling_utils import load_model_dict_into_meta

        unexpected_keys = load_model_dict_into_meta(
            controlnet, diffusers_format_controlnet_checkpoint, dtype=torch_dtype
        )
        if controlnet._keys_to_ignore_on_load_unexpected is not None:
            for pat in controlnet._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint were not used when initializing {controlnet.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        controlnet.load_state_dict(diffusers_format_controlnet_checkpoint)

    if torch_dtype is not None:
        controlnet = controlnet.to(torch_dtype)

    return {"controlnet": controlnet}

def fetch_ldm_config_and_checkpoint(
    pretrained_model_link_or_path,
    class_name,
    original_config_file=None,
    resume_download=False,
    force_download=False,
    proxies=None,
    token=None,
    cache_dir=None,
    local_files_only=None,
    revision=None,
):
    checkpoint = load_single_file_model_checkpoint(
        pretrained_model_link_or_path,
        resume_download=resume_download,
        force_download=force_download,
        proxies=proxies,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
    )
    original_config = fetch_original_config(class_name, checkpoint, original_config_file)

    return original_config, checkpoint


class FromOriginalControlNetMixin:
    """
    Load pretrained ControlNet weights saved in the `.ckpt` or `.safetensors` format into a [`ControlNetModel`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`ControlNetModel`] from pretrained ControlNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            config_file (`str`, *optional*):
                Filepath to the configuration YAML file associated with the model. If not provided it will default to:
                https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v15.yaml
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        Examples:

        ```py
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"  # can also be a local path
        model = ControlNetModel.from_single_file(url)

        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
        pipe = StableDiffusionControlNetPipeline.from_single_file(url, controlnet=controlnet)
        ```
        """
        original_config_file = kwargs.pop("original_config_file", None)
        config_file = kwargs.pop("config_file", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        class_name = cls.__name__
        if (config_file is not None) and (original_config_file is not None):
            raise ValueError(
                "You cannot pass both `config_file` and `original_config_file` to `from_single_file`. Please use only one of these arguments."
            )

        original_config_file = config_file or original_config_file
        original_config, checkpoint = fetch_ldm_config_and_checkpoint(
            pretrained_model_link_or_path=pretrained_model_link_or_path,
            class_name=class_name,
            original_config_file=original_config_file,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        upcast_attention = kwargs.pop("upcast_attention", False)
        image_size = kwargs.pop("image_size", None)

        component = create_diffusers_controlnet_model_from_ldm(
            class_name,
            original_config,
            checkpoint,
            upcast_attention=upcast_attention,
            image_size=image_size,
            torch_dtype=torch_dtype,
        )
        controlnet = component["controlnet"]
        if torch_dtype is not None:
            controlnet = controlnet.to(torch_dtype)

        return controlnet
