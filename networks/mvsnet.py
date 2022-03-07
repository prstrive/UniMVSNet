import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .module import *

Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self, mode="regression"):
        super(DepthNet, self).__init__()
        self.mode = mode
        assert self.mode in ("regression", "classification", "unification"), "Don't support {}!".format(mode)

    def forward(self, cost_reg, depth_values, num_depth, interval, prob_volume_init=None):
        prob_volume_pre = cost_reg.squeeze(1)  # (b, d, h, w)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        if self.mode == "regression":
            prob_volume = F.softmax(prob_volume_pre, dim=1)  # (b, ndepth, h, w)
            depth = depth_regression(prob_volume, depth_values=depth_values)  # (b, h, w)
            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1,
                                                    padding=0).squeeze(1)
                depth_index = depth_regression(prob_volume,
                                               depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                depth_index = depth_index.clamp(min=0, max=num_depth - 1)
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        elif self.mode == "classification":
            prob_volume = F.softmax(prob_volume_pre, dim=1)  # (b, ndepth, h, w)
            depth = winner_take_all(prob_volume, depth_values)  # (b, h, w)
            photometric_confidence, _ = torch.max(prob_volume, dim=1)
        elif self.mode == "unification":
            prob_volume = torch.sigmoid(prob_volume_pre)  # (b, ndepth, h, w)
            depth = unity_regression(prob_volume, depth_values, interval)
            photometric_confidence, _ = torch.max(F.softmax(prob_volume_pre, dim=1), dim=1)
            # photometric_confidence = torch.max(prob_volume, dim=1)[0] / torch.sum(prob_volume, dim=1)
        else:
            raise NotImplementedError("Don't support {}!".format(self.mode))

        return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume,
                "depth_values": depth_values, "interval": interval}


class CostAgg(nn.Module):
    def __init__(self, mode="variance", in_channels=None):
        super(CostAgg, self).__init__()
        self.mode = mode
        assert mode in ("variance", "adaptive"), "Don't support {}!".format(mode)
        if self.mode == "adaptive":
            self.weight_net = nn.ModuleList([AggWeightNetVolume(in_channels[i]) for i in range(len(in_channels))])

    def forward(self, features, proj_matrices, depth_values, stage_idx):
        """
        :param stage_idx: stage
        :param features: [ref_fea, src_fea1, src_fea2, ...], fea shape: (b, c, h, w)
        :param proj_matrices: (b, nview, ...) [ref_proj, src_proj1, src_proj2, ...]
        :param depth_values: (b, ndepth, h, w)
        :return: matching cost volume (b, c, ndepth, h, w)
        """
        ref_feature, src_features = features[0], features[1:]
        proj_matrices = torch.unbind(proj_matrices, 1)  # to list
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        num_views = len(features)
        num_depth = depth_values.shape[1]

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        if self.mode == "variance":
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
            del ref_volume
        elif self.mode == "adaptive":
            volume_adapt = None

        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.mode == "variance":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            elif self.mode == "adaptive":
                # (b, c, d, h, w)
                warped_volume = (ref_volume - warped_volume).pow_(2)
                weight = self.weight_net[stage_idx](warped_volume)
                if volume_adapt is None:
                    volume_adapt = (weight + 1) * warped_volume
                else:
                    volume_adapt = volume_adapt + (weight + 1) * warped_volume

            del warped_volume

        # aggregate multiple feature volumes by variance
        if self.mode == "variance":
            volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            return volume_variance
        elif self.mode == "adaptive":
            return volume_adapt / (num_views - 1)


class MVSNet(nn.Module):
    def __init__(self, ndepths, depth_interval_ratio, cr_base_chs=None, fea_mode="fpn", agg_mode="variance", depth_mode="regression"):
        super(MVSNet, self).__init__()

        if cr_base_chs is None:
            cr_base_chs = [8] * len(ndepths)
        self.ndepths = ndepths
        self.depth_interval_ratio = depth_interval_ratio
        self.fea_mode = fea_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)

        print("netphs:", ndepths)
        print("depth_intervals_ratio:", depth_interval_ratio)
        print("cr_base_chs:", cr_base_chs)
        print("fea_mode:", fea_mode)
        print("agg_mode:", agg_mode)
        print("depth_mode:", depth_mode)

        assert len(ndepths) == len(depth_interval_ratio)

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, mode=self.fea_mode)
        self.cost_aggregation = CostAgg(agg_mode, self.feature.out_channels)

        self.cost_regularization = nn.ModuleList(
            [CostRegNet(in_channels=self.feature.out_channels[i], base_channels=self.cr_base_chs[i]) for i in range(self.num_stage)])

        self.DepthNet = DepthNet(depth_mode)

    def forward(self, imgs, proj_matrices, depth_values):
        """
        :param is_flip: augment only for 3D-UNet
        :param imgs: (b, nview, c, h, w)
        :param proj_matrices:
        :param depth_values:
        :return:
        """
        depth_interval = (depth_values[0, -1] - depth_values[0, 0]) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        ori_shape = imgs[:, 0].shape[2:]  # (H, W)

        outputs = {}
        last_depth = None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            # stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            # stage1: 1/4, stage2: 1/2, stage3: 1
            stage_scale = 2 ** (3 - stage_idx - 1)

            stage_shape = [ori_shape[0] // int(stage_scale), ori_shape[1] // int(stage_scale)]

            if stage_idx == 0:
                last_depth = depth_values
            else:
                last_depth = last_depth.detach()

            # (B, D, H, W)
            depth_range_samples, interval = get_depth_range_samples(last_depth=last_depth,
                                                                    ndepth=self.ndepths[stage_idx],
                                                                    depth_inteval_pixel=self.depth_interval_ratio[
                                                                                            stage_idx] * depth_interval,
                                                                    shape=stage_shape  # only for first stage
                                                                    )

            if stage_idx > 0:
                depth_range_samples = F.interpolate(depth_range_samples, stage_shape, mode='bilinear', align_corners=Align_Corners_Range)

            # (b, c, d, h, w)
            cost_volume = self.cost_aggregation(features_stage, proj_matrices_stage, depth_range_samples, stage_idx)
            # cost volume regularization
            # (b, 1, d, h, w)
            cost_reg = self.cost_regularization[stage_idx](cost_volume)

            # depth
            outputs_stage = self.DepthNet(cost_reg, depth_range_samples, num_depth=self.ndepths[stage_idx], interval=interval)

            last_depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
