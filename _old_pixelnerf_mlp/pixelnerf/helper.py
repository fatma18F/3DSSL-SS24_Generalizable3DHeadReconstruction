conf = {
    "model": {
        "use_encoder": True,
        "use_global_encoder": False,
        "use_xyz": True,
        "canon_xyz": False,
        "use_code": True,
        "code": {
            "num_freqs": 6,
            "freq_factor": 1.5,
            "include_input": True
        },
        "use_viewdirs": True,
        "use_code_viewdirs": False,
        "mlp_coarse": {
            "type": "resnet",
            "n_blocks": 3,
            "d_hidden": 512
        },
        "mlp_fine": {
            "type": "resnet",
            "n_blocks": 3,
            "d_hidden": 512
        },
        "encoder": {
            "backbone": "resnet34",
            "pretrained": True,
            "num_layers": 4
        }
    },
    "renderer": {
        "n_coarse": 64,
        "n_fine": 32,
        "n_fine_depth": 16,
        "depth_std": 0.01,
        "sched": [],
        "white_bkgd": True
    },
    "loss": {
        "rgb": {
            "use_l1": False
        },
        "rgb_fine": {
            "use_l1": False
        },
        "alpha": {
            "lambda_alpha": 0.0,
            "clamp_alpha": 100,
            "init_epoch": 5
        },
        "lambda_coarse": 1.0,
        "lambda_fine": 1.0
    },
    "train": {
        "print_interval": 2,
        "save_interval": 50,
        "vis_interval": 100,
        "eval_interval": 50,
        "accu_grad": 1,
        "num_epoch_repeats": 1
    }
}



