import torch
import numpy as np
from torch import nn
from spconv.pytorch import SparseConvTensor
from .detector3d_template import Detector3DTemplate
from ...utils import common_utils
from ..model_utils.sparse_temporal_fusion import SparseTemporalFusionWithAdapter

# 尝试导入诊断工具，如果失败则禁用诊断
try:
    from ..model_utils.feature_diagnostics import get_diagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    def get_diagnostics():
        return None


# --- 模块2：升级 VoxelNeXt_KP 类 ---
class VoxelNeXt_KP(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # self.module_list 现在包含了 self.vfe, self.backbone_3d, self.dense_head
        self.module_list = self.build_networks()

        # 根据您的配置文件，BEV特征图的通道数是384
        bev_feature_dim = 384
        num_frames = 3  # 时序帧数

        # 使用新的稀疏时序融合模块（包含Adapter）
        # 优势：保持稀疏性、LayerNorm不破坏背景、自动分布对齐
        self.sparse_temporal_fusion = SparseTemporalFusionWithAdapter(
            channels=bev_feature_dim,
            num_frames=num_frames,
            num_heads=8,
            window_size=10,  # BEV平面窗口大小（约1m范围）
            dropout=0.1
        )

    def forward(self, batch_dict):
        # --- 核心修改：重构循环，确保数据字典的完整性 ---

        # 1. 准备工作
        # batch_dict中的值现在是列表，列表长度等于序列长度
        sequence_length = len(batch_dict['frame_id'])
        batch_size = batch_dict['batch_size']

        bev_features_list = []
        processed_frame_dicts = []

        # 2. 循环处理序列中的每一帧
        for t in range(sequence_length):

            frame_batch_dict = {}
            gt_names_temp = None  # 临时存储字符串类型的数据
            frame_id_temp = None

            for key, val in batch_dict.items():
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

            # 3. 让单帧数据流过VFE和3D骨干网络
            frame_batch_dict = self.vfe(frame_batch_dict)
            frame_batch_dict = self.backbone_3d(frame_batch_dict)

            # 4. 收集每一帧的稀疏BEV特征（不转Dense，保持稀疏性）
            # backbone_3d的输出是一个spconv.SparseConvTensor对象
            sparse_tensor = frame_batch_dict['encoded_spconv_tensor']

            bev_features_list.append(sparse_tensor)
            processed_frame_dicts.append(frame_batch_dict)

        # 6. 使用稀疏时序融合模块（保持稀疏性，不破坏背景）
        # 输入：List[SparseConvTensor] (T帧稀疏特征)
        # 输出：SparseConvTensor (融合后的当前帧稀疏特征，已对齐分布)
        fused_sparse_tensor = self.sparse_temporal_fusion(bev_features_list)

        # 清理中间变量
        del bev_features_list
        torch.cuda.empty_cache()

        # [诊断点] 融合后的稀疏张量（用于Dense Head）
        if not self.training and DIAGNOSTICS_AVAILABLE:
            diagnostics = get_diagnostics()
            if diagnostics is not None:
                diagnostics.log_sparse_tensor(fused_sparse_tensor, "sparse_fused_output_to_densehead")

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
            loss, tb_dict, disp_dict = self.get_training_loss(final_batch_dict)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            # 评估模式：final_batch_dict 已经包含了 dense_head 的所有输出
            # 在评估模式下，dense_head 会自动设置 'final_box_dicts'
            pred_dicts, recall_dicts = self.post_processing(final_batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict_for_loss=None):
        # 确保dense_head能获取到正确的dict
        if batch_dict_for_loss:
            self.dense_head.forward_ret_dict.update(batch_dict_for_loss)

        disp_dict = {}
        loss, tb_dict = self.dense_head.get_loss()
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
