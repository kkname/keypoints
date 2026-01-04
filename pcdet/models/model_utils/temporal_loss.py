"""
Temporal Consistency Loss for Multi-Frame Keypoint Estimation
用于多帧关键点估计的时序一致性损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalConsistencyLoss(nn.Module):
    """
    计算跨帧时序一致性损失，利用 object_id 进行跨帧匹配

    损失包括：
    1. Position Smoothness Loss: 相邻帧的关键点位置应该平滑变化
    2. Velocity Consistency Loss: 速度变化应该平滑（加速度小）
    """

    def __init__(self,
                 position_weight=1.0,
                 velocity_weight=0.5,
                 min_matched_frames=2):
        """
        Args:
            position_weight: 位置平滑性损失的权重
            velocity_weight: 速度一致性损失的权重
            min_matched_frames: 最少需要匹配的帧数（少于此数的物体不参与loss计算）
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.min_matched_frames = min_matched_frames

    def forward(self, per_frame_predictions, obj_ids, gt_obj_ids=None):
        """
        计算时序一致性损失

        Args:
            per_frame_predictions: List[Dict], 长度为 T（时间步数）
                每个元素是一帧的预测结果字典，包含：
                - 'pred_dicts': List[Dict], 每个batch的预测
                    - 'pred_kps': (N, 14, 3) 预测的关键点
                    - 'pred_boxes': (N, 7) 预测的边界框
                    - 'pred_labels': (N,) 预测的类别标签
            obj_ids: List[List[array]], shape: [T][B]
                obj_ids[t][b] 是第t帧第b个batch的物体ID数组
                例如：obj_ids[0][0] = array(['person_01', 'person_02'])
            gt_obj_ids: 可选，ground truth的obj_ids（当前实现未使用）

        Returns:
            loss_dict: Dict
                - 'temporal_smoothness_loss': 位置平滑性损失
                - 'temporal_velocity_loss': 速度一致性损失
                - 'temporal_loss': 总损失
        """
        if per_frame_predictions is None or len(per_frame_predictions) < self.min_matched_frames:
            # 如果帧数不足，返回0损失（使用CPU设备）
            return {
                'temporal_smoothness_loss': torch.tensor(0.0),
                'temporal_velocity_loss': torch.tensor(0.0),
                'temporal_loss': torch.tensor(0.0)
            }

        # 步骤1: 跨帧ID匹配
        matched_objects = self.match_objects_across_frames(per_frame_predictions, obj_ids)

        if len(matched_objects) == 0:
            # 没有匹配到跨帧物体
            device = per_frame_predictions[0]['pred_dicts'][0]['pred_kps'].device
            return {
                'temporal_smoothness_loss': torch.tensor(0.0, device=device),
                'temporal_velocity_loss': torch.tensor(0.0, device=device),
                'temporal_loss': torch.tensor(0.0, device=device)
            }

        # 步骤2: 计算位置平滑性损失
        loss_smoothness = self.compute_position_smoothness(matched_objects, per_frame_predictions)

        # 步骤3: 计算速度一致性损失（需要至少3帧）
        if len(per_frame_predictions) >= 3:
            loss_velocity = self.compute_velocity_consistency(matched_objects, per_frame_predictions)
        else:
            loss_velocity = torch.tensor(0.0, device=loss_smoothness.device)

        # 总损失
        total_loss = self.position_weight * loss_smoothness + self.velocity_weight * loss_velocity

        return {
            'temporal_smoothness_loss': loss_smoothness,
            'temporal_velocity_loss': loss_velocity,
            'temporal_loss': total_loss
        }

    def match_objects_across_frames(self, per_frame_predictions, obj_ids):
        """
        根据 object_id 进行跨帧匹配

        Returns:
            matched_objects: Dict[str, Dict]
                key: object_id (例如 'person_01')
                value: {
                    'frame_indices': List[int],     # 该物体出现在哪些帧 [0, 1, 2]
                    'batch_index': int,              # 属于哪个batch
                    'pred_indices_in_gt': List[int], # 在每帧GT中的索引
                }
        """
        matched_objects = {}

        # obj_ids 格式: [T][B]，其中 obj_ids[t][b] 是一个 numpy array
        num_frames = len(obj_ids)
        num_batches = len(obj_ids[0]) if num_frames > 0 else 0

        # 遍历每个batch
        for batch_idx in range(num_batches):
            # 收集该batch在所有帧中的 obj_id
            obj_tracker = {}  # {obj_id: [(frame_idx, gt_idx), ...]}

            for frame_idx in range(num_frames):
                frame_obj_ids = obj_ids[frame_idx][batch_idx]  # numpy array

                if frame_obj_ids is None or len(frame_obj_ids) == 0:
                    continue

                # 遍历当前帧的每个物体
                for gt_idx, obj_id in enumerate(frame_obj_ids):
                    obj_id_str = str(obj_id)

                    if obj_id_str not in obj_tracker:
                        obj_tracker[obj_id_str] = []

                    obj_tracker[obj_id_str].append((frame_idx, gt_idx))

            # 只保留在至少 min_matched_frames 帧中出现的物体
            for obj_id, frame_gt_pairs in obj_tracker.items():
                if len(frame_gt_pairs) >= self.min_matched_frames:
                    matched_objects[f"{batch_idx}_{obj_id}"] = {
                        'batch_index': batch_idx,
                        'frame_indices': [pair[0] for pair in frame_gt_pairs],
                        'pred_indices_in_gt': [pair[1] for pair in frame_gt_pairs],
                        'obj_id': obj_id
                    }

        return matched_objects

    def compute_position_smoothness(self, matched_objects, per_frame_predictions):
        """
        计算位置平滑性损失：相邻帧的关键点位置应该接近

        Loss = smooth_L1( kp_t - kp_{t-1} )
        """
        total_loss = 0.0
        num_pairs = 0

        for obj_key, match_info in matched_objects.items():
            batch_idx = match_info['batch_index']
            frame_indices = match_info['frame_indices']
            gt_indices = match_info['pred_indices_in_gt']

            # 遍历连续的帧对
            for i in range(len(frame_indices) - 1):
                frame_t1 = frame_indices[i]
                frame_t2 = frame_indices[i + 1]
                gt_idx_t1 = gt_indices[i]
                gt_idx_t2 = gt_indices[i + 1]

                # 获取预测的关键点
                # per_frame_predictions[t]['pred_dicts'] 是一个list，包含每个batch的预测
                pred_dict_t1 = per_frame_predictions[frame_t1]['pred_dicts'][batch_idx]
                pred_dict_t2 = per_frame_predictions[frame_t2]['pred_dicts'][batch_idx]

                # 检查预测框数量是否足够
                if gt_idx_t1 >= pred_dict_t1['pred_kps'].shape[0] or \
                   gt_idx_t2 >= pred_dict_t2['pred_kps'].shape[0]:
                    continue

                # 提取关键点 (14, 3)
                kp_t1 = pred_dict_t1['pred_kps'][gt_idx_t1]
                kp_t2 = pred_dict_t2['pred_kps'][gt_idx_t2]

                # 计算平滑性损失（使用 smooth L1 loss，对异常值更鲁棒）
                loss = F.smooth_l1_loss(kp_t1, kp_t2, reduction='mean')
                total_loss += loss
                num_pairs += 1

        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            # 没有有效的帧对
            device = per_frame_predictions[0]['pred_dicts'][0]['pred_kps'].device
            return torch.tensor(0.0, device=device)

    def compute_velocity_consistency(self, matched_objects, per_frame_predictions):
        """
        计算速度一致性损失：速度变化应该平滑（加速度接近0）

        velocity_01 = kp_t1 - kp_t0
        velocity_12 = kp_t2 - kp_t1
        acceleration = velocity_12 - velocity_01
        Loss = mean(acceleration^2)
        """
        total_loss = 0.0
        num_triplets = 0

        for obj_key, match_info in matched_objects.items():
            batch_idx = match_info['batch_index']
            frame_indices = match_info['frame_indices']
            gt_indices = match_info['pred_indices_in_gt']

            # 需要至少3帧才能计算速度一致性
            for i in range(len(frame_indices) - 2):
                frame_t0 = frame_indices[i]
                frame_t1 = frame_indices[i + 1]
                frame_t2 = frame_indices[i + 2]

                gt_idx_t0 = gt_indices[i]
                gt_idx_t1 = gt_indices[i + 1]
                gt_idx_t2 = gt_indices[i + 2]

                # 获取预测字典
                pred_dict_t0 = per_frame_predictions[frame_t0]['pred_dicts'][batch_idx]
                pred_dict_t1 = per_frame_predictions[frame_t1]['pred_dicts'][batch_idx]
                pred_dict_t2 = per_frame_predictions[frame_t2]['pred_dicts'][batch_idx]

                # 检查索引有效性
                if gt_idx_t0 >= pred_dict_t0['pred_kps'].shape[0] or \
                   gt_idx_t1 >= pred_dict_t1['pred_kps'].shape[0] or \
                   gt_idx_t2 >= pred_dict_t2['pred_kps'].shape[0]:
                    continue

                # 提取关键点 (14, 3)
                kp_t0 = pred_dict_t0['pred_kps'][gt_idx_t0]
                kp_t1 = pred_dict_t1['pred_kps'][gt_idx_t1]
                kp_t2 = pred_dict_t2['pred_kps'][gt_idx_t2]

                # 计算速度
                velocity_01 = kp_t1 - kp_t0
                velocity_12 = kp_t2 - kp_t1

                # 计算加速度
                acceleration = velocity_12 - velocity_01

                # 加速度应该接近0（使用L2损失）
                loss = torch.mean(acceleration ** 2)
                total_loss += loss
                num_triplets += 1

        if num_triplets > 0:
            return total_loss / num_triplets
        else:
            device = per_frame_predictions[0]['pred_dicts'][0]['pred_kps'].device
            return torch.tensor(0.0, device=device)
