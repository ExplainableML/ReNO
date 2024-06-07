import torch


def get_optimizer(
    optimizer_name: str, latents: torch.Tensor, lr: float, nesterov: bool
):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([latents], lr=lr, eps=1e-2)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD([latents], lr=lr, nesterov=nesterov, momentum=0.9)
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [latents],
            lr=lr,
            max_iter=10,
            history_size=3,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
    return optimizer
