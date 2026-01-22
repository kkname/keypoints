import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import common_utils, keypoint_coder_utils, loss_utils
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
        kp_mask = forward_ret_dict["batch_gt_of_kp_mask"].view(-1, code_size // 3)
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

        kp_mask_expanded = kp_mask.unsqueeze(-1).repeat(1, 1, 3).view(rcnn_batch_size, -1)
        rcnn_kp_loss_reg = self.reg_kp_loss_func(
            (rcnn_reg_kp.view(rcnn_batch_size, -1) * kp_mask_expanded).unsqueeze(dim=0),
            (reg_targets.view(rcnn_batch_size, -1) * kp_mask_expanded).unsqueeze(dim=0),
        )
        rcnn_kp_loss_reg = rcnn_kp_loss_reg.view(rcnn_batch_size, -1).sum(dim=-1)
        rcnn_kp_loss_reg = (rcnn_kp_loss_reg * fg_mask.float()).sum() / max(fg_sum, 1)
        rcnn_kp_loss_reg = rcnn_kp_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_kp_reg_weight']

        tb_dict = {'rcnn_kp_loss_reg': rcnn_kp_loss_reg.item()}
        return rcnn_kp_loss_reg, tb_dict

    def get_loss(self, tb_dict=None):
        rcnn_loss, tb_dict = super().get_loss(tb_dict=tb_dict)
        rcnn_kp_loss_reg, reg_tb_dict = self.get_kp_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_kp_loss_reg
        tb_dict.update(reg_tb_dict)
        rcnn_kp_vis_loss, vis_tb_dict = self.get_kp_vis_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_kp_vis_loss
        tb_dict.update(vis_tb_dict)
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
        kp_mask = forward_ret_dict["batch_gt_of_kp_mask"].view(-1, code_size // 3)
        gt_kp_vis = forward_ret_dict['batch_gt_of_kp_vis'].view(-1, code_size // 3).float()
        rcnn_reg_kp_vis = forward_ret_dict['rcnn_reg_kp_vis'].view(-1, code_size // 3)

        fg_mask = (reg_valid_mask > 0).float()
        vis_weights = fg_mask.unsqueeze(-1) * kp_mask.float()

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
