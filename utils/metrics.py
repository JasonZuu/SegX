import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy import stats
from sklearn.utils import resample
from tqdm import tqdm
import torch.nn.functional as F
import torch


metric_rules = {"auroc": "max",
                'auprc': 'max',
                'f1': 'max',
                'sensitivity': 'max',
                'specificity': 'max',
                'loss': "min",
                }


def auprc_score_multi_class(y_true, y_score, average="macro"):
    """
    Calculate the area under the precision-recall curve for multi-class classification.

    Parameters:
    - y_true: array-like of shape (n_samples, n_classes)
        True class labels in one-hot encoded format.
    - y_score: array-like of shape (n_samples, n_classes)
        Predicted scores or probabilities for each class.
    - average: str, optional (default="macro")
        Strategy to average the score. Supported values:
        - "macro": Calculate metrics for each label, and find their unweighted mean.
        - "weighted": Calculate metrics for each label, and find their average weighted by support.
    
    Returns:
    - auc_scores: float
        Averaged AUPRC score.
    """
    # Ensure y_true and y_score have the same number of classes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    
    n_classes = y_true.shape[1]

    auc_scores = []
    
    for i in range(n_classes):
        # Get precision and recall for the current class (one-vs-rest)
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        # Compute AUC for this class
        auc_score = auc(recall, precision)
        auc_scores.append(auc_score)
    
    # Return the averaged AUPRC score
    if average == "macro":
        return np.mean(auc_scores)
    elif average == "weighted":
        class_counts = np.sum(y_true, axis=0)  # Sum across samples to get support for each class
        return np.average(auc_scores, weights=class_counts)
    else:
        raise ValueError(f"Unsupported average method: {average}")
    

def auroc_score_multi_class(y_true, y_score, average="macro"):
    """
    Calculate the area under the ROC curve (AUROC) for multi-class classification.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True class labels.
    - y_score: array-like of shape (n_samples, n_classes)
        Predicted scores or probabilities for each class.
    - average: str, optional (default="macro")
        Strategy to average the score. Supported values:
        - "macro": Calculate metrics for each label, and find their unweighted mean.
        - "weighted": Calculate metrics for each label, and find their average weighted by support.
    
    Returns:
    - auc_scores: float
        Averaged AUROC score.
    """
    # Binarize the true labels for one-vs-rest strategy
    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    auc_scores = []
    
    for i in range(n_classes):
        # Get ROC curve for the current class (one-vs-rest)
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        # Compute AUC for this class
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
    
    # Return the averaged AUROC score
    if average == "macro":
        return np.mean(auc_scores)
    elif average == "weighted":
        class_counts = np.sum(y_true_bin, axis=0)
        return np.average(auc_scores, weights=class_counts)
    else:
        raise ValueError(f"Unsupported average method: {average}")


def calculate_p_value(mean1, ci1, n1, mean2, ci2, n2):
    """
    Calculate the p-value for the two-tailed test for two independent samples with the given means, 
    confidence intervals, and sample sizes.
    
    Args:
    - mean1, mean2: Means of the two samples.
    - ci1, ci2: Confidence intervals of the two samples (assuming 95% confidence level).
    - n1, n2: Sample sizes of the two samples.
    
    Returns:
    - p-value for the two-tailed test.
    """
    # Assuming 95% confidence level, t value for 2-tailed test
    df = n1 + n2 - 2
    t_value = stats.t.ppf(1 - 0.025, df)

    # Calculating Standard Error from Confidence Interval
    se1 = ci1 / t_value
    se2 = ci2 / t_value

    # Calculating the combined standard error
    sed = np.sqrt(se1**2.0 + se2**2.0)

    # Calculating the t-statistic
    t_stat = (mean1 - mean2) / sed

    # Calculating the p-value
    p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
    return p


def calculate_improvement(x0, x1, negative=False):
    if negative:
        return (x0 - x1) / x0 * 100
    else:
        return (x1 - x0) / x0 * 100
    

def bootstrap_metrics(y_scores, y_labels, cutoff, n_bootstraps=1000):
    auprc_scores = []
    auroc_scores = []
    f1_scores = []
    loss_scores = []

    pbar = tqdm(total=n_bootstraps, desc="Bootstrapping", unit="bootstrap")
    for _ in range(n_bootstraps):
        # Bootstrap sample indices
        indices = resample(np.arange(len(y_labels)), replace=True)
        boot_y_scores = y_scores[indices]
        boot_y_labels = y_labels[indices]

        # Compute metrics
        auprc = auprc_score_multi_class(boot_y_labels, boot_y_scores)
        auroc = auroc_score_multi_class(boot_y_labels, boot_y_scores)
        loss = cross_entropy_np(boot_y_scores, boot_y_labels)

        y_preds = y_scores > cutoff
        f1 = f1_score(boot_y_labels, y_preds, average="macro")

        auprc_scores.append(auprc)
        auroc_scores.append(auroc)
        f1_scores.append(f1)
        loss_scores.append(loss)

        pbar.update(1)

        # Calculate means and confidence intervals (for example, 95% CI)
    metrics = {
        'auprc': {"mean": np.mean(auprc_scores),
                  "ci": _compute_bootstrap_ci(auprc_scores)},
        'auroc': {"mean": np.mean(auroc_scores),
                  "ci": _compute_bootstrap_ci(auroc_scores)},
        'f1': {"mean": np.mean(f1_scores),
               "ci": _compute_bootstrap_ci(f1_scores)},
        'loss': {"mean": np.mean(loss_scores),
                 "ci": _compute_bootstrap_ci(loss_scores)}}
    return metrics


def get_optimal_f1_threshold(y_true, y_score):
    """
    Calculate the optimal threshold for F1 score for multi-class classification (one-vs-rest strategy).

    Parameters:
    - y_true: array-like of shape (n_samples, n_classes)
        True class labels in one-hot encoded format.
    - y_score: array-like of shape (n_samples, n_classes)
        Predicted scores or probabilities for each class.

    Returns:
    - thresholds: array-like of shape (n_classes,)
        Optimal threshold for F1 score for each class.
    """
    n_classes = y_true.shape[1]
    optimal_thresholds = []

    for i in range(n_classes):
        # Get precision, recall, and thresholds for the current class (one-vs-rest)
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_score[:, i])
        
        # Calculate F1 scores for all thresholds
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add epsilon to avoid division by zero

        # Find the best threshold (which gives the maximum F1 score)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]  # Skip the last threshold, as it's not used for F1 calculation
        optimal_thresholds.append(best_threshold)

    return np.array(optimal_thresholds)


def _compute_bootstrap_ci(data):
    lower = np.percentile(data, 2.5)
    upper = np.percentile(data, 97.5)
    return (lower, upper)


@torch.no_grad()
def cross_entropy_np(y_scores, labels):
    y_scores = torch.tensor(y_scores, device='cpu')
    labels = torch.tensor(labels, dtype=torch.float, device='cpu')
    loss = F.cross_entropy(y_scores, labels)
    return loss.item()
