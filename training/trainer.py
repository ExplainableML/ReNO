import logging
import math
from typing import Dict, List, Optional, Tuple

import PIL
import PIL.Image
import torch
from diffusers import DiffusionPipeline

from rewards import clip_img_transform
from rewards.base_reward import BaseRewardLoss


class LatentNoiseTrainer:
    """Trainer for optimizing latents with reward losses."""

    def __init__(
        self,
        reward_losses: List[BaseRewardLoss],
        model: DiffusionPipeline,
        n_iters: int,
        n_inference_steps: int,
        seed: int,
        no_optim: bool = False,
        regularize: bool = True,
        regularization_weight: float = 0.01,
        grad_clip: float = 0.1,
        log_metrics: bool = True,
        save_all_images: bool = False,
        imageselect: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        self.reward_losses = reward_losses
        self.model = model
        self.n_iters = n_iters
        self.n_inference_steps = n_inference_steps
        self.seed = seed
        self.no_optim = no_optim
        self.regularize = regularize
        self.regularization_weight = regularization_weight
        self.grad_clip = grad_clip
        self.log_metrics = log_metrics
        self.save_all_images = save_all_images
        self.imageselect = imageselect
        self.device = device
        self.preprocess_fn = clip_img_transform(224)

    def train(
        self,
        latents: torch.Tensor,
        prompt: str,
        optimizer: torch.optim.Optimizer,
        save_dir: Optional[str] = None,
        multi_apply_fn=None,
    ) -> Tuple[PIL.Image.Image, Dict[str, float], Dict[str, float]]:
        logging.info(f"Optimizing latents for prompt '{prompt}'.")
        best_loss = torch.inf
        best_image = None
        initial_image = None
        initial_rewards = None
        best_rewards = None
        best_latents = None
        latent_dim = math.prod(latents.shape[1:])
        for iteration in range(self.n_iters):
            to_log = ""
            rewards = {}
            optimizer.zero_grad()
            generator = torch.Generator("cuda").manual_seed(self.seed)
            if self.imageselect:
                new_latents = torch.randn_like(
                    latents, device=self.device, dtype=latents.dtype
                )
                image = self.model.apply(
                    new_latents,
                    prompt,
                    generator=generator,
                    num_inference_steps=self.n_inference_steps,
                )
            else:
                image = self.model.apply(
                    latents=latents,
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=self.n_inference_steps,
                )
            if initial_image is None:
                if multi_apply_fn is not None:
                    initial_image = multi_apply_fn(latents.detach(), prompt)
                else:
                    initial_image = image
                image_numpy = (
                    initial_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                )
                initial_image = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
            if self.no_optim:
                best_image = image
                break

            total_loss = 0
            preprocessed_image = self.preprocess_fn(image)
            for reward_loss in self.reward_losses:
                loss = reward_loss(preprocessed_image, prompt)
                to_log += f"{reward_loss.name}: {loss.item():.4f}, "
                total_loss += loss * reward_loss.weighting
                rewards[reward_loss.name] = loss.item()
            rewards["total"] = total_loss.item()
            to_log += f"Total: {total_loss.item():.4f}"
            total_reward_loss = total_loss.item()
            if self.regularize:
                # compute in fp32 to avoid overflow
                latent_norm = torch.linalg.vector_norm(latents).to(torch.float32)
                log_norm = torch.log(latent_norm)
                regularization = self.regularization_weight * (
                    0.5 * latent_norm**2 - (latent_dim - 1) * log_norm
                )
                to_log += f", Latent norm: {latent_norm.item()}"
                rewards["norm"] = latent_norm.item()
                total_loss += regularization.to(total_loss.dtype)
            if self.log_metrics:
                logging.info(f"Iteration {iteration}: {to_log}")
            if total_reward_loss < best_loss:
                best_loss = total_reward_loss
                best_image = image
                best_rewards = rewards
                best_latents = latents.detach().cpu()
            if iteration != self.n_iters - 1 and not self.imageselect:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(latents, self.grad_clip)
                optimizer.step()
            if self.save_all_images:
                image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
                image_pil.save(f"{save_dir}/{iteration}.png")
            if initial_rewards is None:
                initial_rewards = rewards
        image_numpy = best_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        best_image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
        if multi_apply_fn is not None:
            multi_step_image = multi_apply_fn(best_latents.to("cuda"), prompt)
            image_numpy = (
                multi_step_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            )
            best_image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
        return initial_image, best_image_pil, initial_rewards, best_rewards
