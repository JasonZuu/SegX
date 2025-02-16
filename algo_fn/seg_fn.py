import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import torch.nn.functional as f

from models import UNet
from datasets import get_dataloader
from configs.train_configs import SegmentationConfig
from models.tracker import PerformanceTracker
from algo_fn.test_fn import seg_test_fn


def seg_train_fn(model: UNet, train_config: SegmentationConfig,
                train_dataset, val_dataset, write_log=True):
    print(f"Start Training on Seed {train_config.seed}")
    model.to(train_config.device)

    tracker = PerformanceTracker(early_stop_epochs=train_config.early_stop_epochs)

    train_loader = get_dataloader(train_dataset, mode="original",
                                  batch_size=train_config.batch_size,
                                  num_classes=train_config.num_classes)

    val_loader = get_dataloader(val_dataset, mode="original",
                                batch_size=train_config.batch_size,
                                num_classes=train_config.num_classes)

    if write_log:
        log_dir = train_config.log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    model_optim = torch.optim.Adam(model.parameters(), lr=train_config.model_lr)
    model_optim_scheduler = torch.optim.lr_scheduler.StepLR(model_optim,
                                                            step_size=1,
                                                            gamma=train_config.lr_decay_gamma)

    device = train_config.device

    epoch_num = train_config.epoch_num

    for i_epoch in range(epoch_num):
        _train_loop(model, model_optim,
                   loader=train_loader, device=device, writer=writer, current_epoch=i_epoch)
        metric_dict = seg_test_fn(model, loader=val_loader, train_config=train_config)
        model_state_dict = {"model": deepcopy(model.state_dict())}

        if writer is not None:
            for key, value in metric_dict.items():
                writer.add_scalar(f"val/{key}", value, i_epoch)

        continue_flag = tracker.update(metric_dict, model_state_dict)
        if not continue_flag:
            break

        if i_epoch >= train_config.warmup_epochs:
            model_optim_scheduler.step()

    state_dict = tracker.export_best_model_state_dict()
    best_val_metric_dict = tracker.export_best_metric_dict()
    model.load_state_dict(state_dict["model"])

    # write log
    if write_log:
        torch.save(state_dict, f"{log_dir}/model.pth")
        writer.close()

    return best_val_metric_dict


def _train_loop(model: UNet, model_optim,
               loader, writer: SummaryWriter, device: str, current_epoch: int):
    print("Training Loop")
    model.train()

    num_step = current_epoch * len(loader)

    pbar = tqdm(total=len(loader), desc=f"Epoch {current_epoch}")
    epoch_loss = 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = f.binary_cross_entropy(outputs, masks)

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()
        lr = model_optim.param_groups[0]['lr']

        epoch_loss += loss.item()

        # write log
        if writer is not None:
            log_dict = {"seg_loss": loss.item(),
                        "lr": lr}
            for key, value in log_dict.items():
                writer.add_scalar(f"train/{key}", value, num_step)

        num_step += 1
        pbar.update()
        pbar.set_postfix({"loss":loss.item()})
