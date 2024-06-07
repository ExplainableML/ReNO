import os

import clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from rewards.base_reward import BaseRewardLoss


class AestheticLoss(BaseRewardLoss):
    """CLIP reward loss function for optimization."""

    def __init__(
        self,
        weigthing: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        self.clip_model, self.preprocess_fn = clip.load(
            "ViT-L/14", device=device, download_root=cache_dir
        )
        self.clip_model = self.clip_model.to(device, dtype=dtype)
        self.mlp = MLP(768).to(device, dtype=dtype)
        s = torch.load(
            f"{os.getcwd()}/ckpts/aesthetic-model.pth"
        )  # load the model you trained previously or the model available in this repo
        self.mlp.load_state_dict(s)
        self.clip_model.eval()
        if memsave:
            import memsave_torch.nn

            self.mlp = memsave_torch.nn.convert_to_memory_saving(self.mlp)
            self.clip_model = memsave_torch.nn.convert_to_memory_saving(
                self.clip_model
            ).to(device, dtype=dtype)

        self.freeze_parameters(self.clip_model.parameters())
        self.freeze_parameters(self.mlp.parameters())
        super().__init__("Aesthetic", weigthing)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda"):
            clip_img_features = self.clip_model.encode_image(image)
            l2 = torch.norm(clip_img_features, p=2, dim=-1, keepdim=True)
            l2 = torch.where(
                l2 == 0,
                torch.tensor(
                    1.0, device=clip_img_features.device, dtype=clip_img_features.dtype
                ),
                l2,
            )
            clip_img_features = clip_img_features / l2
        return clip_img_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        return None

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        return None

    def __call__(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        if self.memsave:
            image = image.to(torch.float32)
        image_features = self.get_image_features(image)

        image_features_normed = self.process_features(image_features.to(torch.float16))

        aesthetic_loss = 10.0 - self.mlp(image_features_normed).mean()
        return aesthetic_loss


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
