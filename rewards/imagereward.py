import ImageReward as RM
import torch

from rewards.base_reward import BaseRewardLoss


class ImageRewardLoss:
    """Image reward loss for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        self.name = "ImageReward"
        self.weighting = weighting
        self.dtype = dtype
        self.imagereward_model = RM.load("ImageReward-v1.0", download_root=cache_dir)
        self.imagereward_model = self.imagereward_model.to(
            device=device, dtype=self.dtype
        )
        self.imagereward_model.eval()
        BaseRewardLoss.freeze_parameters(self.imagereward_model.parameters())

    def __call__(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        imagereward_score = self.score_diff(prompt, image)
        return (2 - imagereward_score).mean()

    def score_diff(self, prompt, image):
        # text encode
        text_input = self.imagereward_model.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.imagereward_model.device)
        image_embeds = self.imagereward_model.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.imagereward_model.device
        )
        text_output = self.imagereward_model.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].to(
            self.imagereward_model.device, dtype=self.dtype
        )
        rewards = self.imagereward_model.mlp(txt_features)
        rewards = (rewards - self.imagereward_model.mean) / self.imagereward_model.std

        return rewards
