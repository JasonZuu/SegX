import torch
import numpy as np
from sklearn.metrics import f1_score

from datasets import get_dataloader
from utils.metrics import bootstrap_metrics, get_optimal_f1_threshold
from utils.metrics import auroc_score_multi_class, auprc_score_multi_class, cross_entropy_np


@torch.no_grad()
def cls_test_fn(model, train_config, val_dataset, test_dataset):
    print(f"Start Testing on Seed {train_config.seed}")
    model.to(train_config.device)
    model.eval()
    
    val_loader = get_dataloader(val_dataset, mode="original",
                                batch_size=train_config.batch_size,
                                num_classes=train_config.num_classes)
    test_loader = get_dataloader(test_dataset, mode="original",
                                 batch_size=train_config.batch_size,
                                 num_classes=train_config.num_classes)
    
    # val
    metric_dict = _cls_test_loop(model, loader=val_loader, device=train_config.device)
    cutoff = metric_dict["cutoff"]

    # test
    metric_dict = _cls_test_loop(model, loader=test_loader, device=train_config.device,
                             cutoff=cutoff, run_bootstrap=True)
                
    return metric_dict


@torch.no_grad()
def _cls_test_loop(model, loader, device, run_bootstrap=False, cutoff=None):
    model.eval()

    # collections
    y_scores_collection = []
    labels_collection = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        y_scores, hidden = model.forward_with_hidden(images, use_sigmoid=True)

        # eval
        batch_y_score = y_scores.detach().cpu().numpy()
        batch_labels = labels.cpu().numpy().astype(np.int8)

        y_scores_collection.append(batch_y_score)
        labels_collection.append(batch_labels)
    
    y_scores = np.concatenate(y_scores_collection)
    labels = np.concatenate(labels_collection)

    auprc = auprc_score_multi_class(y_true=labels, y_score=y_scores)
    auroc = auroc_score_multi_class(y_true=labels, y_score=y_scores)

    if cutoff is None:
        cutoff = get_optimal_f1_threshold(y_true=labels, y_score=y_scores)

    y_preds = y_scores > cutoff
    f1 = f1_score(y_true=labels, y_pred=y_preds, average="macro")
    loss = cross_entropy_np(y_scores, labels)

    if run_bootstrap:
        bootstrap_results = bootstrap_metrics(y_scores, labels, cutoff, n_bootstraps=1000)
        metric_dict = {"auprc": {"mean": auprc,
                                 "bootstrap_mean": bootstrap_results["auprc"]["mean"],
                                 "lower_ci": bootstrap_results["auprc"]["ci"][0],
                                "upper_ci": bootstrap_results["auprc"]["ci"][1],},
                        "auroc": {"mean": auroc,
                                 "bootstrap_mean": bootstrap_results["auroc"]["mean"],
                                 "lower_ci": bootstrap_results["auroc"]["ci"][0],
                                "upper_ci": bootstrap_results["auroc"]["ci"][1],},
                        "f1": {"mean": f1,
                                 "bootstrap_mean": bootstrap_results["f1"]["mean"],
                                 "lower_ci": bootstrap_results["f1"]["ci"][0],
                                "upper_ci": bootstrap_results["f1"]["ci"][1],},
                        "loss": {"mean": loss,
                                 "bootstrap_mean": bootstrap_results["loss"]["mean"],
                                 "lower_ci": bootstrap_results["loss"]["ci"][0],
                                 "upper_ci": bootstrap_results["loss"]["ci"][1],}}

    else:
        metric_dict = {"auprc": auprc,
                        "auroc": auroc,
                        "f1": f1,
                        'cutoff': cutoff,
                        "loss": loss}

    return metric_dict


@torch.no_grad()
def seg_test_fn(model, train_config, loader):  # will enable bootstrap later
    device = train_config.device
    model.eval().to(device)
    dice_scores = []
    iou_scores = []
    losses = []
    cutoff = 0.5

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = (outputs > cutoff).float()  

        # Dice
        dice = (2 * (preds * masks).sum()) / ((preds + masks).sum() + 1e-6)
        dice_scores.append(dice.item())

        # IoU
        intersection = (preds * masks).sum().item()  # 计算交集
        union = preds.sum().item() + masks.sum().item() - intersection  # 计算并集
        iou = intersection / (union + 1e-6)  # 避免分母为0
        iou_scores.append(iou)

        # binary cross entropy
        bce = torch.nn.functional.binary_cross_entropy(outputs, masks)
        losses.append(bce.item())

    dice_mean = np.mean(dice_scores)
    iou_mean = np.mean(iou_scores)
    loss_mean = np.mean(losses)
    metric_dict = {"dice": dice_mean,
                   "iou": iou_mean,
                   "loss": loss_mean}

    return metric_dict