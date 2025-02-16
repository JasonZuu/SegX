import argparse
import numpy as np
import pandas as pd
import os

from models import ResNetBased, DenseNetBased, UNet
from configs.train_configs import ClassificationConfig, SegmentationConfig
from datasets.loader import init_cls_datasets, init_seg_datasets
from algo_fn import cls_train_fn, seg_train_fn
from utils.misc import set_random_seed, get_log_dir
from utils.hparams import get_db_storage_path, load_best_hparams


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset', type=str, default='chestx', choices=["ham", "chestx"], help='dataset')
    parser.add_argument("--task", type=str, default="seg", choices=["cls", "seg"])
    parser.add_argument("--model", type=str, default="unet",choices=["resnet", "densenet", 'unet'])
    parser.add_argument("--pretrained", action="store_true", help="whether to use pretrained model")
    parser.add_argument("--use_best_hparams", action="store_true", help="whether to use best hparams")
    parser.add_argument('--db_dir', type=str, default='optuna_db', help='db_save_dir')
    parser.add_argument("--log_dir", type=str, default="log", help="log_dir")
    parser.add_argument("--seed", type=int, default=18087, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_random_seed(args.seed)

    log_dir = get_log_dir(log_dir=args.log_dir, model=args.model, dataset=args.dataset, task=args.task)
    db_path = get_db_storage_path(args)


    # init config, datasets, and model
    if args.task == "cls":
        dataset_configs, train_dataset, val_dataset, _ = init_cls_datasets(args)
        train_config = ClassificationConfig()
        train_config.seed = args.seed
        train_config.log_dir = log_dir
        train_config.num_classes = dataset_configs.num_classes
    elif args.task == "seg":
        dataset_configs, train_dataset, val_dataset, _ = init_seg_datasets(args)
        train_config = SegmentationConfig()
        train_config.seed = args.seed
        train_config.log_dir = log_dir
        train_config.num_classes = 1  # binary segmentation
    else:
        raise ValueError(f"Invalid task {args.task}")

    if args.use_best_hparams:
        train_config = load_best_hparams(args.model, db_path, train_config)

    if args.model == "resnet":
        assert args.task == "cls", "ResNet is only for classification task"
        model = ResNetBased(num_classes=train_config.num_classes, pretrained=args.pretrained)
    elif args.model == "densenet":
        assert args.task == "cls", "DenseNet is only for classification task"
        model = DenseNetBased(num_classes=train_config.num_classes, pretrained=args.pretrained)
    elif args.model == "unet":
        assert args.task == "seg", "UNet is only for segmentation task"
        model = UNet(out_channels=train_config.num_classes)
    else:
        raise ValueError(f"Invalid model {args.model}")
    
    # train
    if args.task == "cls":
        val_result = cls_train_fn(model, train_config,
                        train_dataset=train_dataset, val_dataset=val_dataset,
                        write_log=True)
        metrics = list(val_result.keys())
        val_result_df = pd.DataFrame({"metric": metrics,
                                      "value": [val_result[metric] for metric in metrics]})
        val_result_df.to_csv(f"{log_dir}/val_result.csv", index=False)
    elif args.task == "seg":
        val_result = seg_train_fn(model, train_config,
                                train_dataset=train_dataset, val_dataset=val_dataset,
                                write_log=True)
        metrics = list(val_result.keys())
        val_result_df = pd.DataFrame({"metric": metrics,
                                      "value": [val_result[metric] for metric in metrics]})