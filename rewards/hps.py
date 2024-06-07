import huggingface_hub
import torch
from hpsv2.src.open_clip import create_model, get_tokenizer

from rewards.base_reward import BaseRewardLoss


class HPSLoss(BaseRewardLoss):
    """HPS reward loss function for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        self.hps_model = create_model(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=dtype,
            device=device,
            cache_dir=cache_dir,
        )
        checkpoint_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", "HPS_v2.1_compressed.pt", cache_dir=cache_dir
        )
        self.hps_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)["state_dict"]
        )
        self.hps_tokenizer = get_tokenizer("ViT-H-14")
        if memsave:
            import memsave_torch.nn

            self.hps_model = memsave_torch.nn.convert_to_memory_saving(self.hps_model)
        self.hps_model = self.hps_model.to(device, dtype=dtype)
        self.hps_model.eval()
        self.freeze_parameters(self.hps_model.parameters())
        super().__init__("HPS", weighting)
        self.hps_model.set_grad_checkpointing(True)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        hps_image_features = self.hps_model.encode_image(image)
        return hps_image_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        hps_text = self.hps_tokenizer(prompt).to("cuda")
        hps_text_features = self.hps_model.encode_text(hps_text)
        return hps_text_features

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        logits_per_image = image_features @ text_features.T
        hps_loss = 1 - torch.diagonal(logits_per_image)[0]
        return hps_loss
