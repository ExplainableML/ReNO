import logging

import torch
from diffusers import (AutoencoderKL, DDPMScheduler,
                       EulerAncestralDiscreteScheduler, LCMScheduler,
                       Transformer2DModel, UNet2DConditionModel)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from models.RewardPixart import RewardPixartPipeline, freeze_params
from models.RewardStableDiffusion import RewardStableDiffusion
from models.RewardStableDiffusionXL import RewardStableDiffusionXL


def get_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    cache_dir: str,
    memsave: bool = False,
):
    logging.info(f"Loading model: {model_name}")
    if model_name == "sd-turbo":
        pipe = RewardStableDiffusion.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=dtype,
            variant="fp16",
            cache_dir=cache_dir,
            memsave=memsave,
        )
        pipe = pipe.to(device, dtype)
    elif model_name == "sdxl-turbo":
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        pipe = RewardStableDiffusionXL.from_pretrained(
            "stabilityai/sdxl-turbo",
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            cache_dir=cache_dir,
            memsave=memsave,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        pipe = pipe.to(device, dtype)
    elif model_name == "pixart":
        pipe = RewardPixartPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=dtype,
            cache_dir=cache_dir,
            memsave=memsave,
        )
        pipe.transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512",
            subfolder="transformer",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        pipe.scheduler = DDPMScheduler.from_pretrained(
            "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512",
            subfolder="scheduler",
            cache_dir=cache_dir,
        )

        # speed-up T5
        pipe.text_encoder.to_bettertransformer()
        pipe.transformer.eval()
        freeze_params(pipe.transformer.parameters())
        pipe.transformer.enable_gradient_checkpointing()
        pipe = pipe.to(device)
    elif model_name == "hyper-sd":
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"
        # Load model.
        unet = UNet2DConditionModel.from_config(
            base_model_id, subfolder="unet", cache_dir=cache_dir
        ).to(device, dtype)
        unet.load_state_dict(
            load_file(
                hf_hub_download(repo_name, ckpt_name, cache_dir=cache_dir),
                device="cuda",
            )
        )
        pipe = RewardStableDiffusionXL.from_pretrained(
            base_model_id,
            unet=unet,
            torch_dtype=dtype,
            variant="fp16",
            cache_dir=cache_dir,
            is_hyper=True,
            memsave=memsave,
        )
        # Use LCM scheduler instead of ddim scheduler to support specific timestep number inputs
        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config, cache_dir=cache_dir
        )
        pipe = pipe.to(device, dtype)
        # upcast vae
        pipe.vae = pipe.vae.to(dtype=torch.float32)
        # pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return pipe
