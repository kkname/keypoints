import torch
import numpy as np
from torch import nn
from spconv.pytorch import SparseConvTensor
from .detector3d_template import Detector3DTemplate
from ...utils import common_utils

# 尝试导入诊断工具，如果失败则禁用诊断
try:
    from ..model_utils.feature_diagnostics import get_diagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    def get_diagnostics():
        return None


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * 2,  # 减小FFN
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_embedding = nn.Parameter(torch.randn(1, 10, input_dim))

    def forward(self, x):
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:, :seq_len, :]
        return self.transformer_encoder(x)


# --- 模块2：升级 VoxelNeXt_KP 类 ---
class VoxelNeXt_KP(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # self.module_list 现在包含了 self.vfe, self.backbone_3d, self.dense_head
        self.module_list = self.build_networks()

        # 根据您的配置文件，BEV特征图的通道数是384
        bev_feature_dim = 384
        # 启用混合精度后，可以使用更强的Transformer
        self.temporal_fusion_module = TemporalTransformer(
            input_dim=bev_feature_dim, num_heads=8, num_layers=2, dropout=0.1
        )

        # 添加特征适配层：将Transformer输出的特征分布对齐到预训练dense_head期望的分布
        # 使用1×1卷积，初始化为恒等映射，让模型自己学习需要的调整
        self.feature_adapter = nn.Conv2d(bev_feature_dim, bev_feature_dim, kernel_size=1, bias=True)

        # 正确初始化为恒等变换：weight shape是(out_c, in_c, 1, 1)
        with torch.no_grad():
            # 创建单位矩阵并reshape为4D: (384, 384) -> (384, 384, 1, 1)
            identity = torch.eye(bev_feature_dim).unsqueeze(-1).unsqueeze(-1)
            self.feature_adapter.weight.copy_(identity)
            self.feature_adapter.bias.zero_()

    def forward(self, batch_dict):
        # --- 核心修改：重构循环，确保数据字典的完整性 ---

        # 1. 准备工作
        # batch_dict中的值现在是列表，列表长度等于序列长度
        sequence_length = len(batch_dict['frame_id'])
        batch_size = batch_dict['batch_size']

        # DEBUG: 检查原始batch_dict的结构（已验证，注释掉以减少日志输出）
        # print(f"\n[DEBUG] Original batch_dict structure:")
        # print(f"  sequence_length: {sequence_length}")
        # print(f"  batch_size: {batch_size}")
        # print(f"  batch_dict keys: {list(batch_dict.keys())}")

        bev_features_list = []
        processed_frame_dicts = []
        per_frame_predictions = []  # 新增：存储每帧的预测结果

        # 2. 循环处理序列中的每一帧
        for t in range(sequence_length):

            frame_batch_dict = {}
            gt_names_temp = None  # 临时存储字符串类型的数据
            frame_id_temp = None

            for key, val in batch_dict.items():
                # obj_ids 只用于 temporal loss，不需要传给 dense_head
                if key == 'obj_ids':
                    continue

                # 对于时序数据（列表格式且长度等于序列长度），提取第t帧
                if isinstance(val, list) and len(val) == sequence_length:
                    extracted_val = val[t]
                    # gt_names 和 frame_id 是字符串类型，需要特殊处理
                    if key == 'gt_names':
                        gt_names_temp = extracted_val
                    elif key == 'frame_id':
                        frame_id_temp = extracted_val
                    else:
                        frame_batch_dict[key] = extracted_val
                # 对于时序数据（numpy数组格式，第二维是时间维度）
                elif isinstance(val, np.ndarray) and val.ndim >= 2 and val.shape[1] == sequence_length:
                    frame_batch_dict[key] = val[:, t]
                # 对于时序数据（torch.Tensor格式，第二维是时间维度）- 关键修复！
                elif isinstance(val, torch.Tensor) and val.ndim >= 2 and val.shape[1] == sequence_length:
                    frame_batch_dict[key] = val[:, t]
                # 对于非时序数据（batch_size等标量），直接复制
                else:
                    frame_batch_dict[key] = val

            # 先将数值数据送到GPU
            common_utils.load_data_to_gpu(frame_batch_dict)

            # 再添加字符串类型的数据（不需要送GPU）
            if gt_names_temp is not None:
                frame_batch_dict['gt_names'] = gt_names_temp
            if frame_id_temp is not None:
                frame_batch_dict['frame_id'] = frame_id_temp

            # DEBUG: 已验证数据提取正确，注释掉以减少日志输出
            # if t == 0:
            #     print(f"\n[DEBUG] Frame {t} data after extraction:")
            #     print(f"  gt_boxes shape: {frame_batch_dict.get('gt_boxes').shape if 'gt_boxes' in frame_batch_dict else 'NOT FOUND'}")

            # 3. 让单帧数据流过VFE和3D骨干网络 (完全复用现有模块)
            # 这两个模块会直接修改 frame_batch_dict
            frame_batch_dict = self.vfe(frame_batch_dict)
            frame_batch_dict = self.backbone_3d(frame_batch_dict)
            # 4. 在训练时，对每一帧都做预测（用于计算时序一致性损失）
            # 注意：当temporal_weight=0时，禁用此部分以节省显存
            # if self.training:
            #     # 调用 dense_head 前向传播获取原始预测
            #     temp_dict = frame_batch_dict.copy()
            #     temp_dict = self.dense_head(temp_dict)
            #
            #     # 手动调用 generate_predicted_boxes 来解码预测（避免触发 reorder_rois_for_refining）
            #     pred_dicts_raw = self.dense_head.forward_ret_dict['pred_dicts']
            #     voxel_indices = self.dense_head.forward_ret_dict['voxel_indices']
            #
            #     # 从 sparse tensor 中提取 spatial_shape
            #     sparse_tensor = frame_batch_dict['encoded_spconv_tensor']
            #     spatial_shape = sparse_tensor.spatial_shape
            #
            #     # 手动解码预测
            #     pred_dicts_decoded = self.dense_head.generate_predicted_boxes(
            #         batch_size=batch_size,
            #         pred_dicts=pred_dicts_raw,
            #         voxel_indices=voxel_indices,
            #         spatial_shape=spatial_shape
            #     )
            #
            #     # 保存解码后的预测结果（包含 pred_kps）
            #     per_frame_predictions.append({
            #         'pred_dicts': pred_dicts_decoded,
            #         'batch_size': batch_size
            #     })
            #
            #     # 清理临时变量和缓存
            #     del temp_dict, pred_dicts_raw, pred_dicts_decoded, voxel_indices
            #     torch.cuda.empty_cache()

            # 5. 收集每一帧的BEV特征图
            # backbone_3d的输出是一个spconv.SparseConvTensor对象
            sparse_tensor = frame_batch_dict['encoded_spconv_tensor']
            # .dense()方法会将其转换为 (B, C, Z, Y, X) 的密集张量
            # 我们通过[0]取出Z维度，得到 (B, C, Y, X) 的BEV特征图
            bev_feature_map = sparse_tensor.dense()
            frame_batch_dict['spatial_features_2d'] = bev_feature_map

            bev_features_list.append(bev_feature_map)
            processed_frame_dicts.append(frame_batch_dict)

        # 5. 将收集到的BEV特征图列表堆叠成一个序列张量 (B, T, C, H, W)
        bev_features_sequence = torch.stack(bev_features_list, dim=1)

        # 清理BEV特征列表释放显存
        del bev_features_list
        torch.cuda.empty_cache()

        # 6. 通过Temporal Transformer融合多帧特征
        # bev_features_sequence: (B, T, C, H, W)
        B, T, C, H, W = bev_features_sequence.shape

        # 将空间维度展平: (B, T, C, H, W) -> (B, T, H*W, C)
        bev_features_flat = bev_features_sequence.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

        # 为了节省显存，分块处理空间位置
        chunk_size = 64  # 每次处理64个空间位置
        num_chunks = (H * W + chunk_size - 1) // chunk_size
        fused_features_chunks = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, H * W)

            # 提取当前chunk: (B, T, chunk_size, C)
            chunk_features = bev_features_flat[:, :, start_idx:end_idx, :]

            # 重塑为 (B * chunk_size, T, C) 以便Transformer处理
            B_chunk, T_chunk, spatial_chunk, C_chunk = chunk_features.shape
            chunk_features_reshaped = chunk_features.permute(0, 2, 1, 3).reshape(B_chunk * spatial_chunk, T_chunk, C_chunk)

            # 通过Transformer: (B*chunk_size, T, C) -> (B*chunk_size, T, C)
            chunk_fused = self.temporal_fusion_module(chunk_features_reshaped)

            # 提取最后一帧的融合特征: (B*chunk_size, C)
            chunk_fused_last = chunk_fused[:, -1, :]

            # 重塑回 (B, chunk_size, C)
            chunk_fused_last = chunk_fused_last.reshape(B_chunk, spatial_chunk, C_chunk)

            fused_features_chunks.append(chunk_fused_last)

        # 合并所有chunks: (B, H*W, C)
        fused_features_flat = torch.cat(fused_features_chunks, dim=1)

        # 重塑回BEV格式: (B, H*W, C) -> (B, C, H, W)
        fused_bev_feature = fused_features_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 清理中间变量
        del bev_features_sequence, bev_features_flat, fused_features_chunks, fused_features_flat
        torch.cuda.empty_cache()

        # [诊断点1] Transformer输出（adapter之前）
        if not self.training and DIAGNOSTICS_AVAILABLE:
            diagnostics = get_diagnostics()
            if diagnostics is not None:
                diagnostics.log_dense_tensor(fused_bev_feature, "transformer_output_before_adapter")

        # 7. 应用特征适配层，对齐特征分布
        fused_bev_feature = self.feature_adapter(fused_bev_feature)

        # [诊断点2] Adapter输出
        if not self.training and DIAGNOSTICS_AVAILABLE:
            diagnostics = get_diagnostics()
            if diagnostics is not None:
                diagnostics.log_dense_tensor(fused_bev_feature, "adapter_output")

        # 8. 将融合后的BEV特征注入到最后一帧的SparseConvTensor中
        # 获取最后一帧的 sparse tensor 作为模板
        last_frame_dict = processed_frame_dicts[-1]
        sparse_tensor_template = last_frame_dict['encoded_spconv_tensor']

        # 从 dense BEV 特征中提取对应 sparse 位置的特征
        # sparse_tensor_template.indices 的格式是 [batch_idx, y, x] (BEV只有2D空间)
        # 我们需要根据这些索引从 fused_bev_feature 中提取特征
        # 注意：BEV 特征形状是 (B, C, H, W)，而 indices 是 (N, 3)
        batch_indices = sparse_tensor_template.indices[:, 0].long()
        spatial_y = sparse_tensor_template.indices[:, 1].long()  # H 维度
        spatial_x = sparse_tensor_template.indices[:, 2].long()  # W 维度

        # 从 fused_bev_feature 中提取对应位置的特征
        fused_features_sparse = fused_bev_feature[batch_indices, :, spatial_y, spatial_x]  # (N, C)

        # 创建新的 SparseConvTensor，使用融合后的特征
        fused_sparse_tensor = SparseConvTensor(
            features=fused_features_sparse,
            indices=sparse_tensor_template.indices,
            spatial_shape=sparse_tensor_template.spatial_shape,
            batch_size=batch_size
        )

        # [诊断点3] 重建后的Sparse Tensor
        if not self.training and DIAGNOSTICS_AVAILABLE:
            diagnostics = get_diagnostics()
            if diagnostics is not None:
                diagnostics.log_sparse_tensor(fused_sparse_tensor, "fused_sparse_tensor_to_densehead")

        # 9. 准备最终送入预测头的数据字典（提取最后一帧的数据）
        final_batch_dict = {}
        for key, val in batch_dict.items():
            if isinstance(val, list) and len(val) == sequence_length:
                # 列表类型（时序数据），提取最后一帧
                final_batch_dict[key] = val[-1]
            elif isinstance(val, torch.Tensor) and val.ndim >= 2 and val.shape[1] == sequence_length:
                # Tensor类型且第二维是时间维度，提取所有batch的最后一帧
                final_batch_dict[key] = val[:, -1]
            else:
                # 其他情况（非时序数据或标量），直接使用
                final_batch_dict[key] = val

        final_batch_dict['batch_size'] = batch_size
        final_batch_dict['encoded_spconv_tensor'] = fused_sparse_tensor

        final_batch_dict = self.dense_head(final_batch_dict)
        # ----------------------------------------------------

        if self.training:
            # 提取 obj_ids（如果存在）
            obj_ids = batch_dict.get('obj_ids', None)
            loss, tb_dict, disp_dict = self.get_training_loss(
                final_batch_dict,
                per_frame_predictions=per_frame_predictions,
                obj_ids=obj_ids
            )
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            # 评估模式：final_batch_dict 已经包含了 dense_head 的所有输出
            # 在评估模式下，dense_head 会自动设置 'final_box_dicts'
            pred_dicts, recall_dicts = self.post_processing(final_batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict_for_loss=None, per_frame_predictions=None, obj_ids=None):
        # 确保dense_head能获取到正确的dict
        if batch_dict_for_loss:
            self.dense_head.forward_ret_dict.update(batch_dict_for_loss)

        disp_dict = {}
        # 传递多帧预测结果和 obj_ids 给 dense_head.get_loss()
        loss, tb_dict = self.dense_head.get_loss(
            per_frame_predictions=per_frame_predictions,
            obj_ids=obj_ids
        )
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
