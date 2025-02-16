import torch
import torch.nn.functional as F
from copy import deepcopy


def cross_entropy_loss_fn(logits, targets, reduction='mean'):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(targets * log_probs, dim=-1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def focal_loss_fn(logits, targets, alpha=0.25, gamma=2.0):
    # Calculate the cross entropy loss
    ce_loss = cross_entropy_loss_fn(logits, targets, reduction=None)
    # Calculate the probabilities of the targets
    p_t = torch.exp(-ce_loss)
    # Calculate the focal loss
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
    return focal_loss.mean()


def l1_loss_fn(outputs, targets):
    loss = F.l1_loss(outputs, targets, reduction='mean')
    return loss


def l2_loss_fn(outputs, targets):
    loss = F.mse_loss(outputs, targets, reduction='mean')
    return loss


def cos_loss_fn(outputs, targets, is_allzero=None):
    batch_size = outputs.size(0)
    loss = 1 - F.cosine_similarity(outputs.view(batch_size, -1), targets.view(batch_size, -1), dim=-1)

    if is_allzero is not None:
        batch_size = torch.sum(~is_allzero, dim=-1)
    
    if batch_size == 0:
        return torch.tensor(0.0)
    else:
        return loss.sum() / batch_size


def dp_loss_fn(logits, sensitives, labels=None):
    # Initialize loss to 0
    dp_loss = 0.0

    # Calculate the total number of classes from the sensitives one-hot encoding
    num_groups = sensitives.size(1)

    # Compute loss for each class separately
    for i in range(num_groups):
        # Create a mask for the current class
        group_mask = sensitives[:, i].type(torch.bool)

        group_logits = logits[group_mask]
        group_sensitives = sensitives[group_mask]

        # Avoid computation if there are no elements in the class to prevent NaN loss
        if group_logits.nelement() != 0:
            group_loss = cross_entropy_loss_fn(group_logits, group_sensitives)
            dp_loss = dp_loss + group_loss / group_sensitives.size(0)

    dp_loss = num_groups - dp_loss
    return dp_loss


def eo_loss_fn(logits, sensitives, labels):
    eo_loss = 0.0

    num_sens = sensitives.size(1)
    num_labels = labels.size(1)
    # Compute loss for each class separately
    for i in range(num_sens):
        for j in range(num_labels):
            group_mask = sensitives[:, i].type(torch.bool) & labels[:, j].type(torch.bool)

            group_logits = logits[group_mask]
            group_sensitives = sensitives[group_mask]

            # Avoid computation if there are no elements in the class to prevent NaN loss
            if group_logits.nelement() != 0:
                group_loss = cross_entropy_loss_fn(group_logits, group_sensitives)
                eo_loss = eo_loss + group_loss / group_sensitives.size(0)

    eo_loss = num_sens - eo_loss
    return eo_loss


def ce_adv_loss_fn(logits, sensitives, labels=None):
    loss = cross_entropy_loss_fn(logits, sensitives)
    return loss


def col_orth_loss_fn(feature_t, feature_a, label, i_epoch, U,
                     conditional=True, margin=0.0, threshold=0.99, moving_epoch=3): 
    
    if conditional:
        indices1 = torch.where(label == 1)[0]
        indices0 = torch.where(label == 0)[0]

        feature_t_0 = torch.index_select(feature_t, 0, indices0)
        feature_t_1 = torch.index_select(feature_t, 0, indices1)
        t_list = [feature_t_0, feature_t_1]

        assert feature_a is not None
        feature_a_0 = torch.index_select(feature_a, 0, indices0)
        feature_a_1 = torch.index_select(feature_a, 0, indices1)
        a_list = [feature_a_0, feature_a_1]
    else:
        t_list = [feature_t]
        a_list = [feature_a]

    loss = 0

    for i in range(len(t_list)):
        feature_t_sub = t_list[i]
        feature_a_sub = a_list[i]
        
        if i_epoch <= moving_epoch:  # update bases
            if U[i] is None:
                U[i] = _update_space(feature_a_sub, threshold)
            else:
                U[i] = _update_space(feature_a_sub, threshold, U[i])

        U_sub = U[i]

        proj_fea = torch.matmul(feature_t_sub, U_sub.to(feature_t.device))
        con_loss = torch.clamp(torch.sum(proj_fea**2) - margin, min=0)

        loss = loss + con_loss / feature_t_sub.shape[0]

    loss = loss / len(t_list)
    U = [_U_sub.detach().requires_grad_() for _U_sub in U]
    return loss, U


def _update_space(feature, threshold, bases=None):
    # Update logic if bases are provided
    if bases is None:
        U, S, _ = torch.linalg.svd(feature.T, full_matrices=False)
        sval_ratio = (S**2) / (S**2).sum()
        r = (torch.cumsum(sval_ratio, -1) < threshold).sum()
        return U[:, :r]

    R2 = torch.matmul(feature.T, feature)
    delta = []
    for ki in range(bases.shape[1]):
        base = bases[:, ki : ki + 1]
        delta_i = torch.matmul(torch.matmul(base.T, R2), base).squeeze()
        delta.append(delta_i)

    delta = torch.hstack(delta)

    _, S_, _ = torch.linalg.svd(feature.T, full_matrices=False)
    sval_total = (S_**2).sum()

    projection_diff = feature - torch.matmul(torch.matmul(bases, bases.T), feature.T).T
    U, S, V = torch.linalg.svd(projection_diff.T, full_matrices=False)

    stack = torch.hstack((delta, S**2))
    S_new, sel_index = torch.topk(stack, len(stack))

    r = 0
    accumulated_sval = 0
    for i in range(len(stack)):
        if accumulated_sval < threshold * sval_total and r < feature.shape[1]:
            accumulated_sval = accumulated_sval + S_new[i].item()
            r = r + 1
        else:
            break

    sel_index = sel_index[:r]
    S_new = S_new[:r]

    Ui = torch.hstack([bases, U])
    U_new = torch.index_select(Ui, 1, sel_index)

    return U_new


def row_orth_loss_fn(feature_t, feature_a, label,
                     conditional=True, margin=0.0):
    # Apply gradient reversal on input embeddings
    if conditional:
        feature_t_0, feature_t_1, feature_s_0, feature_s_1 = _split(feature_t, feature_a, label)
        t_list = [feature_t_0, feature_t_1]
        s_list = [feature_s_0, feature_s_1]
    else:
        t_list = [feature_t]
        s_list = [feature_a]

    loss = 0.0
    for feature_t_sub, feature_a_sub in zip(t_list, s_list):
        feature_t_sub = feature_t_sub - feature_t_sub.mean(0, keepdim=True)
        feature_a_sub = feature_a_sub - feature_a_sub.mean(0, keepdim=True)

        sigma = torch.matmul(feature_t_sub.T, feature_a_sub.detach())

        sigma_loss = torch.clamp(torch.sum(sigma**2) - margin, min=0)
        loss = loss + sigma_loss / sigma.numel()

    loss = loss / len(t_list)

    return loss


def _split(feature_t, feature_s, label):
    indices1 = torch.where(label == 1)[0]
    indices0 = torch.where(label == 0)[0]

    feature_t_0 = torch.index_select(feature_t, 0, indices0)
    feature_t_1 = torch.index_select(feature_t, 0, indices1)

    feature_s_0 = torch.index_select(feature_s, 0, indices0)
    feature_s_1 = torch.index_select(feature_s, 0, indices1)

    return feature_t_0, feature_t_1, feature_s_0, feature_s_1


if __name__ == "__main__":
    logits = torch.randn(10, 2)

    # targets
    bool_targets = torch.rand(10, 1) > 0.5
    targets = bool_targets.long()
    inverse_targets = 1 - targets
    targets = torch.cat((targets, inverse_targets), dim=1)

    # sens
    bool_sensitives = torch.rand(10, 1) > 0.5
    sensitives = bool_targets.long()
    inverse_sensitives = 1 - sensitives
    sensitives = torch.cat((sensitives, inverse_sensitives), dim=1)

    eo_loss = eo_loss_fn(logits, sensitives, targets)
    print(1)
