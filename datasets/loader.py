from torch.utils.data import DataLoader, Dataset

from configs.dataset_configs import HAMDatasetConfig, ChestXDatasetConfig
from datasets import HAMDataset, ChestXDataset
from .sampler import RandomLabelSampler, BalancedLabelSampler


def get_dataloader(dataset: HAMDataset, mode: str,
                   num_classes: int, batch_size: int,
                   num_workers: int=4):
    if mode == "balanced-label":
        sampler = BalancedLabelSampler(labels=dataset.get_labels(),
                                       num_classes=num_classes,
                                       batch_size=batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=num_workers)
    elif mode == "random":
        sampler = RandomLabelSampler(labels=dataset.get_labels(),
                                     num_classes=num_classes,
                                     batch_size=batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=num_workers)
    elif mode == "original":
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")
    return loader


def init_cls_datasets(args):
    # init and update dataset configs
    if args.dataset == "ham":
        dataset_configs = HAMDatasetConfig()
        dataset_configs.seed = args.seed
        train_dataset = HAMDataset(config=dataset_configs, set_type="train")
        val_dataset = HAMDataset(config=dataset_configs, set_type="val")
        test_dataset = HAMDataset(config=dataset_configs, set_type="test")
    elif args.dataset == "chestx":
        dataset_configs = ChestXDatasetConfig()
        dataset_configs.seed = args.seed
        train_dataset = ChestXDataset(config=dataset_configs, set_type="train")
        val_dataset = ChestXDataset(config=dataset_configs, set_type="val")
        test_dataset = ChestXDataset(config=dataset_configs, set_type="test")
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    return dataset_configs, train_dataset, val_dataset, test_dataset


def init_seg_datasets(args):
    # init and update dataset configs
    if args.dataset == "ham":
        dataset_configs = HAMDatasetConfig()
        dataset_configs.seed = args.seed
        train_dataset = HAMDataset(config=dataset_configs, set_type="train", load_mask=True)
        val_dataset = HAMDataset(config=dataset_configs, set_type="val", load_mask=True)
        test_dataset = HAMDataset(config=dataset_configs, set_type="test", load_mask=True)
    elif args.dataset == "chestx":
        dataset_configs = ChestXDatasetConfig()
        dataset_configs.seed = args.seed
        train_dataset = ChestXDataset(config=dataset_configs, set_type="train", load_mask=True)
        val_dataset = ChestXDataset(config=dataset_configs, set_type="val", load_mask=True)
        test_dataset = ChestXDataset(config=dataset_configs, set_type="test", load_mask=True)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    return dataset_configs, train_dataset, val_dataset, test_dataset
