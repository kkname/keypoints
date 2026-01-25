import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import common_utils, keypoint_coder_utils, loss_utils, loss_utils_kp
from .voxelrcnn_head import VoxelRCNNHead


class VoxelRCNNHeadKP(VoxelRCNNHead):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(
            backbone_channels=backbone_channels,
            model_cfg=model_cfg,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            num_class=num_class,
            **kwargs
        )

        self.kp_coder = getattr(keypoint_coder_utils, self.model_cfg.TARGET_CONFIG.KEYPOINT_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('KEYPOINT_CODER_CONFIG', {})
        )

        self.kp_pred_layer = nn.Linear(
            self.reg_pred_layer.in_features,
            self.kp_coder.code_size * self.num_class,
            bias=True
        )
        nn.init.normal_(self.kp_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.kp_pred_layer.bias, 0)

        self.kp_vis_pred_layer = nn.Linear(
            self.reg_pred_layer.in_features,
            self.kp_coder.code_size // 3 * self.num_class,
            bias=True
        )
        nn.init.normal_(self.kp_vis_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.kp_vis_pred_layer.bias, 0)

        self.kp_bone_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('rcnn_kp_bone_weight', 0.5)
        self.kp_bone_half_visible_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get(
            'rcnn_kp_bone_half_visible_weight', 0.5
        )
        self.kp_bone_none_visible_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get(
            'rcnn_kp_bone_none_visible_weight', 0.1
        )
        self.kp_bone_loss_func = loss_utils_kp.BoneLoss(
            [0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 13, 1, 2],
            [13, 3, 4, 5, 6, 9, 10, 11, 12, 1, 2, 7, 8],
            half_visible_weight=self.kp_bone_half_visible_weight,
            none_visible_weight=self.kp_bone_none_visible_weight
        )

    def build_losses(self, losses_cfg):
        super().build_losses(losses_cfg)
        if losses_cfg.REG_LOSS_KP == 'smooth-l1':
            self.add_module(
                'reg_kp_loss_func',
                loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights_kp'])
            )
        else:
            raise NotImplementedError

    def get_kp_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.kp_coder.code_size
        box_code_size = self.box_coder.code_size

        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_kp3d_ct = forward_ret_dict['gt_of_kp'][..., 0:code_size]
        kp_valid = forward_ret_dict["batch_gt_of_kp_mask"].view(-1, code_size // 3)
        gt_kp_vis = forward_ret_dict['batch_gt_of_kp_vis'].view(-1, code_size // 3)
        rcnn_reg_kp = forward_ret_dict['rcnn_reg_kp']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = forward_ret_dict['gt_of_rois'][..., 0:box_code_size].view(-1, box_code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        rois_anchor = roi_boxes3d.clone().detach().view(-1, box_code_size)
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        reg_targets = self.kp_coder.encode_torch(
            gt_kp3d_ct.view(rcnn_batch_size, code_size // 3, 3), rois_anchor
        )

        num_kp = code_size // 3
        pred_kp = rcnn_reg_kp.view(rcnn_batch_size, num_kp, 3)
        target_kp = reg_targets.view(rcnn_batch_size, num_kp, 3)

        vis = gt_kp_vis.float()
        valid = kp_valid.float()
        inv = (1.0 - gt_kp_vis.float()) * kp_valid.float()

        beta_vis = loss_cfgs.LOSS_WEIGHTS.get('rcnn_kp_beta_visible', 0.2)
        beta_inv = loss_cfgs.LOSS_WEIGHTS.get('rcnn_kp_beta_invisible', 1.0)
        lam_inv = loss_cfgs.LOSS_WEIGHTS.get('rcnn_kp_lambda_invisible', 0.4)

        fg = fg_mask.float().unsqueeze(-1).unsqueeze(-1)
        loss_vis_elem = F.smooth_l1_loss(pred_kp, target_kp, reduction='none', beta=beta_vis) * fg * vis.unsqueeze(-1)
        loss_inv_elem = F.smooth_l1_loss(pred_kp, target_kp, reduction='none', beta=beta_inv) * fg * inv.unsqueeze(-1)

        denom_vis = (fg_mask.float().unsqueeze(-1) * vis).sum(dim=0).clamp(min=1.0)
        denom_inv = (fg_mask.float().unsqueeze(-1) * inv).sum(dim=0).clamp(min=1.0)

        loss_vis_vec = loss_vis_elem.sum(dim=(0, 2)) / denom_vis
        loss_inv_vec = loss_inv_elem.sum(dim=(0, 2)) / denom_inv

        weights = loss_vis_vec.new_tensor(loss_cfgs.LOSS_WEIGHTS['code_weights_kp']).view(num_kp, 3).mean(dim=-1)
        loss_vis = (loss_vis_vec * weights).sum() / weights.sum().clamp(min=1e-6)
        loss_inv = (loss_inv_vec * weights).sum() / weights.sum().clamp(min=1e-6)

        rcnn_kp_loss_reg = (loss_vis + lam_inv * loss_inv) * loss_cfgs.LOSS_WEIGHTS['rcnn_kp_reg_weight']

        tb_dict = {'rcnn_kp_loss_reg': rcnn_kp_loss_reg.item()}
        return rcnn_kp_loss_reg, tb_dict

    def get_kp_bone_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.kp_coder.code_size
        box_code_size = self.box_coder.code_size

        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_kp3d_ct = forward_ret_dict['gt_of_kp'][..., 0:code_size]
        gt_kp_vis = forward_ret_dict['batch_gt_of_kp_vis'].view(-1, code_size // 3)
        kp_valid = forward_ret_dict["batch_gt_of_kp_mask"].view(-1, code_size // 3)
        rcnn_reg_kp = forward_ret_dict['rcnn_reg_kp']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = forward_ret_dict['gt_of_rois'][..., 0:box_code_size].view(-1, box_code_size).shape[0]

        rois_anchor = roi_boxes3d.clone().detach().view(-1, box_code_size)
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        reg_targets = self.kp_coder.encode_torch(
            gt_kp3d_ct.view(rcnn_batch_size, code_size // 3, 3), rois_anchor
        )

        num_kp = code_size // 3
        pred_kp = rcnn_reg_kp.view(rcnn_batch_size, num_kp, 3)
        target_kp = reg_targets.view(rcnn_batch_size, num_kp, 3)
        vis = gt_kp_vis.float() * kp_valid.float()

        fg_mask = (reg_valid_mask > 0).float()
        if fg_mask.sum() == 0:
            bone_loss = pred_kp.new_zeros(())
        else:
            bone_parents = pred_kp.new_tensor(self.kp_bone_loss_func.bone_joints_order, dtype=torch.long)
            if self.kp_bone_loss_func.joints_order is None:
                bone_children = torch.arange(pred_kp.size(1), device=pred_kp.device, dtype=torch.long)
            else:
                bone_children = pred_kp.new_tensor(self.kp_bone_loss_func.joints_order, dtype=torch.long)

            pred_u = pred_kp[:, bone_children]
            pred_v = pred_kp[:, bone_parents]
            target_u = target_kp[:, bone_children]
            target_v = target_kp[:, bone_parents]

            vis_u = vis[:, bone_children]
            vis_v = vis[:, bone_parents]
            valid_u = valid[:, bone_children]
            valid_v = valid[:, bone_parents]

            edge_valid = valid_u * valid_v
            both_vis = vis_u * vis_v
            one_vis = ((vis_u + vis_v) == 1).float()
            both_invis = ((vis_u == 0) & (vis_v == 0)).float()

            edge_vis_weight = (
                both_vis
                + self.kp_bone_half_visible_weight * one_vis
                + self.kp_bone_none_visible_weight * both_invis
            )

            edge_weight = fg_mask.unsqueeze(-1) * edge_valid * edge_vis_weight
            if edge_weight.sum() == 0:
                bone_loss = pred_kp.new_zeros(())
            else:
                bone_beta = loss_cfgs.LOSS_WEIGHTS.get('rcnn_kp_bone_beta', 1.0)
                edge_loss = F.smooth_l1_loss(
                    pred_u - pred_v, target_u - target_v, reduction='none', beta=bone_beta
                ).mean(dim=-1)
                bone_loss = (edge_loss * edge_weight).sum() / edge_weight.sum().clamp(min=1.0)
                bone_loss = bone_loss * self.kp_bone_weight
        tb_dict = {'rcnn_kp_bone_loss': bone_loss.item()}
        return bone_loss, tb_dict

    def get_loss(self, tb_dict=None):
        rcnn_loss, tb_dict = super().get_loss(tb_dict=tb_dict)
        rcnn_kp_loss_reg, reg_tb_dict = self.get_kp_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_kp_loss_reg
        tb_dict.update(reg_tb_dict)
        rcnn_kp_vis_loss, vis_tb_dict = self.get_kp_vis_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_kp_vis_loss
        tb_dict.update(vis_tb_dict)
        rcnn_kp_bone_loss, bone_tb_dict = self.get_kp_bone_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_kp_bone_loss
        tb_dict.update(bone_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_keypoints(self, batch_size, rois, kp_preds):
        code_size = self.kp_coder.code_size
        batch_kp_preds = kp_preds.view(batch_size, -1, code_size // 3, 3)

        roi_xyz = rois[:, :, 0:3]
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_kp_preds = self.kp_coder.decode_torch(batch_kp_preds, local_rois)
        batch_kp_preds[..., 0:3] += roi_xyz.unsqueeze(-2)
        return batch_kp_preds

    def get_kp_vis_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.kp_coder.code_size

        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        kp_valid = forward_ret_dict["batch_gt_of_kp_mask"].view(-1, code_size // 3)
        gt_kp_vis = forward_ret_dict['batch_gt_of_kp_vis'].view(-1, code_size // 3).float()
        rcnn_reg_kp_vis = forward_ret_dict['rcnn_reg_kp_vis'].view(-1, code_size // 3)

        fg_mask = (reg_valid_mask > 0).float()
        vis_weights = fg_mask.unsqueeze(-1) * kp_valid.float()

        loss_raw = F.binary_cross_entropy_with_logits(rcnn_reg_kp_vis, gt_kp_vis, reduction='none')
        loss_raw = loss_raw * vis_weights
        denom = torch.clamp(vis_weights.sum(), min=1.0)
        rcnn_kp_vis_loss = loss_raw.sum() / denom
        rcnn_kp_vis_loss = rcnn_kp_vis_loss * loss_cfgs.LOSS_WEIGHTS.get('rcnn_kp_vis_weight', 1.0)

        tb_dict = {'rcnn_kp_vis_loss': rcnn_kp_vis_loss.item()}
        return rcnn_kp_vis_loss, tb_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)
        rois = targets_dict['rois']
        gt_of_rois = targets_dict['gt_of_rois']
        gt_of_kp = targets_dict['gt_of_kp']
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        targets_dict['gt_of_kp_src'] = gt_of_kp.clone().detach()

        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)
        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois

        gt_of_kp[:, :, :, 0:3] = gt_of_kp[:, :, :, 0:3] - roi_center[..., None, :]
        targets_dict['gt_of_kp'] = gt_of_kp

        return targets_dict

    def forward(self, batch_dict):
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        pooled_features = self.roi_grid_pool(batch_dict)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)

        cls_features = self.cls_fc_layers(shared_features)
        reg_features = self.reg_fc_layers(shared_features)

        rcnn_cls = self.cls_pred_layer(cls_features)
        rcnn_reg = self.reg_pred_layer(reg_features)
        rcnn_reg_kp = self.kp_pred_layer(reg_features)
        rcnn_reg_kp_vis = self.kp_vis_pred_layer(reg_features)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_kp_preds = self.generate_predicted_keypoints(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], kp_preds=rcnn_reg_kp
            )
            batch_kp_vis_preds = torch.sigmoid(
                rcnn_reg_kp_vis.view(batch_dict['batch_size'], -1, rcnn_reg_kp_vis.shape[-1])
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['batch_kp_preds'] = batch_kp_preds
            batch_dict['batch_kp_vis_preds'] = batch_kp_vis_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['rcnn_reg_kp'] = rcnn_reg_kp
            targets_dict['rcnn_reg_kp_vis'] = rcnn_reg_kp_vis
            self.forward_ret_dict = targets_dict

        return batch_dict
