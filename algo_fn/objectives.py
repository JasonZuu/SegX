from algo_fn import cls_train_fn, seg_train_fn
from algo_fn import cls_test_fn, seg_test_fn
from configs.train_configs import ClassificationConfig, SegmentationConfig
from models import ResNetBased, DenseNetBased, UNet
from datasets.loader import init_cls_datasets, init_seg_datasets
from datasets import get_dataloader


def cls_objective(trial, args):
    # search space
    model_lr = trial.suggest_float('model_lr', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 8, 64)


    # init and update train configs
    train_config = ClassificationConfig()

    train_config.model_lr = model_lr
    train_config.batch_size = batch_size

    # init and update dataset configs
    dataset_config, train_dataset, val_dataset, test_dataset = init_cls_datasets(args)
    train_config.num_classes = dataset_config.num_classes

    if args.model == "resnet":
        model = ResNetBased(num_classes=train_config.num_classes, pretrained=args.pretrained)
    elif args.model == "densenet":
        model = DenseNetBased(num_classes=train_config.num_classes, pretrained=args.pretrained)
    else:
        raise ValueError(f"Invalid model {args.model}")
    
    cls_train_fn(model, train_config,
                      train_dataset=train_dataset,
                      val_dataset=val_dataset,
                      write_log=False)
    metric_dict = cls_test_fn(model, train_config,
                          val_dataset=val_dataset,
                          test_dataset=test_dataset)
    for metric, value in metric_dict.items():
        trial.set_user_attr(f'test:{metric}', float(value["mean"]))

    loss = metric_dict["loss"]["mean"]

    return loss


def seg_objective(trial, args):
    # search space
    model_lr = trial.suggest_float('model_lr', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 8, 64)

    # init and update train configs
    train_config = SegmentationConfig()

    train_config.model_lr = model_lr
    train_config.batch_size = batch_size

    # init and update dataset configs
    dataset_config, train_dataset, val_dataset, test_dataset = init_seg_datasets(args)
    train_config.num_classes = 1

    if args.model == "unet":
        assert args.task == "seg", "UNet is only for segmentation task"
        model = UNet(out_channels=train_config.num_classes)
    else:
        raise ValueError(f"Invalid model {args.model}")
    
    seg_train_fn(model, train_config,
                 train_dataset=train_dataset,
                 val_dataset=val_dataset,
                 write_log=False)
    test_loader = get_dataloader(test_dataset, mode="original",
                                 batch_size=train_config.batch_size,
                                 num_classes=train_config.num_classes)
    metric_dict = seg_test_fn(model, train_config=train_config, loader=test_loader)
    for metric, value in metric_dict.items():
        trial.set_user_attr(f'test:{metric}', float(value))

    loss = metric_dict["loss"]

    return loss
