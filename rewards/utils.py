from typing import Any, List

import torch
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize)
from transformers import AutoProcessor

from rewards.aesthetic import AestheticLoss
from rewards.base_reward import BaseRewardLoss
from rewards.clip import CLIPLoss
from rewards.hps import HPSLoss
from rewards.imagereward import ImageRewardLoss
from rewards.pickscore import PickScoreLoss


def get_reward_losses(
    args: Any, dtype: torch.dtype, device: torch.device, cache_dir: str
) -> List[BaseRewardLoss]:
    if args.enable_clip or args.enable_pickscore:
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
        )
    reward_losses = []
    if args.enable_hps:
        reward_losses.append(
            HPSLoss(args.hps_weighting, dtype, device, cache_dir, memsave=args.memsave)
        )
    if args.enable_imagereward:
        reward_losses.append(
            ImageRewardLoss(
                args.imagereward_weighting,
                dtype,
                device,
                cache_dir,
                memsave=args.memsave,
            )
        )
    if args.enable_clip:
        reward_losses.append(
            CLIPLoss(
                args.clip_weighting,
                dtype,
                device,
                cache_dir,
                tokenizer,
                memsave=args.memsave,
            )
        )
    if args.enable_pickscore:
        reward_losses.append(
            PickScoreLoss(
                args.pickscore_weighting,
                dtype,
                device,
                cache_dir,
                tokenizer,
                memsave=args.memsave,
            )
        )
    if args.enable_aesthetic:
        reward_losses.append(
            AestheticLoss(
                args.aesthetic_weighting, dtype, device, cache_dir, memsave=args.memsave
            )
        )
    return reward_losses


def clip_img_transform(size: int = 224):
    return Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
