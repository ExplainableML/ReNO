import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process Reward Optimization.")

    # update paths here!
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="HF cache directory",
        default="/shared-local/aoq951/HF_CACHE/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save images",
        default="/shared-local/aoq951/ReNO/outputs",
    )

    # model and optim
    parser.add_argument("--model", type=str, help="Model to use", default="sdxl-turbo")
    parser.add_argument("--lr", type=float, help="Learning rate", default=5.0)
    parser.add_argument("--n_iters", type=int, help="Number of iterations", default=50)
    parser.add_argument(
        "--n_inference_steps", type=int, help="Number of iterations", default=1
    )
    parser.add_argument(
        "--optim",
        choices=["sgd", "adam", "lbfgs"],
        default="sgd",
        help="Optimizer to be used",
    )
    parser.add_argument("--nesterov", default=True, action="store_false")
    parser.add_argument(
        "--grad_clip", type=float, help="Gradient clipping", default=0.1
    )
    parser.add_argument("--seed", type=int, help="Seed to use", default=0)

    # reward losses
    parser.add_argument(
        "--disable_hps", default=True, action="store_false", dest="enable_hps"
    )
    parser.add_argument(
        "--hps_weighting", type=float, help="Weighting for HPS", default=5.0
    )
    parser.add_argument(
        "--disable_imagereward",
        default=True,
        action="store_false",
        dest="enable_imagereward",
    )
    parser.add_argument(
        "--imagereward_weighting",
        type=float,
        help="Weighting for ImageReward",
        default=1.0,
    )
    parser.add_argument(
        "--disable_clip", default=True, action="store_false", dest="enable_clip"
    )
    parser.add_argument(
        "--clip_weighting", type=float, help="Weighting for CLIP", default=0.01
    )
    parser.add_argument(
        "--disable_pickscore",
        default=True,
        action="store_false",
        dest="enable_pickscore",
    )
    parser.add_argument(
        "--pickscore_weighting",
        type=float,
        help="Weighting for PickScore",
        default=0.05,
    )
    parser.add_argument(
        "--disable_aesthetic",
        default=False,
        action="store_false",
        dest="enable_aesthetic",
    )
    parser.add_argument(
        "--aesthetic_weighting",
        type=float,
        help="Weighting for Aesthetic",
        default=0.0,
    )
    parser.add_argument(
        "--disable_reg", default=True, action="store_false", dest="enable_reg"
    )
    parser.add_argument(
        "--reg_weight", type=float, help="Regularization weight", default=0.01
    )

    # task specific
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run",
        default="single",
        choices=[
            "t2i-compbench",
            "single",
            "parti-prompts",
            "geneval",
            "example-prompts",
        ],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to run",
        default="A red dog and a green cat",
    )
    parser.add_argument(
        "--benchmark_reward",
        help="Reward to benchmark on",
        default="total",
        choices=["ImageReward", "PickScore", "HPS", "CLIP", "total"],
    )

    # general
    parser.add_argument("--save_all_images", default=False, action="store_true")
    parser.add_argument("--no_optim", default=False, action="store_true")
    parser.add_argument("--imageselect", default=False, action="store_true")
    parser.add_argument("--memsave", default=False, action="store_true")
    parser.add_argument("--dtype", type=str, help="Data type to use", default="float16")
    parser.add_argument("--device_id", type=str, help="Device ID to use", default=None)
    parser.add_argument(
        "--cpu_offloading",
        help="Enable CPU offloading",
        default=False,
        action="store_true",
    )

    # optional multi-step model
    parser.add_argument("--enable_multi_apply", default=False, action="store_true")
    parser.add_argument(
        "--multi_step_model", type=str, help="Model to use", default="flux"
    )

    args = parser.parse_args()
    return args
