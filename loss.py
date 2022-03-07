import torch
import torch.nn.functional as F


def mvs_loss(inputs, depth_gt_ms, mask_ms, mode, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", [1.0 for k in inputs.keys() if "stage" in k])
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]  # (b, d, h, w)
        depth_est = stage_inputs["depth"]  # (b, h, w)
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        interval = stage_inputs["interval"]  # float
        depth_gt = depth_gt_ms[stage_key]  # (b, h, w)
        mask = mask_ms[stage_key]
        # mask = mask * (depth_hypotheses[:, 0] <= depth_gt).float() * (depth_hypotheses[:, -1] >= depth_gt).float()
        mask = mask > 0.5

        stage_idx = int(stage_key.replace("stage", "")) - 1
        stage_weight = depth_loss_weights[stage_idx]

        if mode == "regression":
            loss = regression_loss(depth_est, depth_gt, mask, stage_weight)
            total_loss += loss
        elif mode == "classification":
            loss = classification_loss(prob_volume, depth_values, interval, depth_gt, mask, stage_weight)
            total_loss += loss
        elif mode == "unification":
            fl_gamas = [2, 1, 0]
            fl_alphas = [0.75, 0.5, 0.25]
            gamma = fl_gamas[stage_idx]
            alpha = fl_alphas[stage_idx]
            loss = unified_focal_loss(prob_volume, depth_values, interval, depth_gt, mask, stage_weight, gamma, alpha)
            total_loss += loss
        else:
            raise NotImplementedError("Only support regression, classification and unification!")

    return total_loss


def regression_loss(depth_est, depth_gt, mask, weight):
    loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
    loss = loss * weight
    return loss


def classification_loss(prob_volume, depth_values, interval, depth_gt, mask, weight):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = (
            ((depth_values - interval / 2) <= depth_gt_volume).float() * ((depth_values + interval / 2) > depth_gt_volume).float())

    NEAR_0 = 1e-4  # Prevent overflow
    prob_volume = torch.where(prob_volume <= 0.0, torch.zeros_like(prob_volume) + NEAR_0, prob_volume)

    loss = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1)[mask].mean()
    loss = loss * weight
    return loss


def unified_focal_loss(prob_volume, depth_values, interval, depth_gt, mask, weight, gamma, alpha):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = ((depth_values <= depth_gt_volume) * ((depth_values + interval) > depth_gt_volume))

    gt_unity_index_volume = torch.zeros_like(prob_volume, requires_grad=False)
    gt_unity_index_volume[gt_index_volume] = 1.0 - (depth_gt_volume[gt_index_volume] - depth_values[gt_index_volume]) / interval

    gt_unity, _ = torch.max(gt_unity_index_volume, dim=1, keepdim=True)
    gt_unity = torch.where(gt_unity > 0.0, gt_unity, torch.ones_like(gt_unity))  # (b, 1, h, w)
    pos_weight = (sigmoid((gt_unity - prob_volume).abs() / gt_unity, base=5) - 0.5) * 4 + 1  # [1, 3]
    neg_weight = (sigmoid(prob_volume / gt_unity, base=5) - 0.5) * 2  # [0, 1]
    focal_weight = pos_weight.pow(gamma) * (gt_unity_index_volume > 0.0).float() + alpha * neg_weight.pow(gamma) * (
            gt_unity_index_volume <= 0.0).float()

    mask = mask.unsqueeze(1).expand_as(depth_values).float()
    loss = (F.binary_cross_entropy(prob_volume, gt_unity_index_volume, reduction="none") * focal_weight * mask).sum() / mask.sum()
    loss = loss * weight
    return loss


def sigmoid(x, base=2.71828):
    return 1 / (1 + torch.pow(base, -x))