from dataclasses import dataclass


@dataclass
class ClsSearchSpaceConfigs:
    search_space = {
        "model_lr": [1e-5, 1e-4, 1e-3],
        "batch_size": [16, 32, 64],
    }


@dataclass
class SegSearchSpaceConfigs:
    search_space = {
        "model_lr": [1e-5, 1e-4, 1e-3],
        "batch_size": [16, 32, 64],
    }