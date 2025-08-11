import torch
from torch import nn
from .detector3d_template import Detector3DTemplate
from ...utils import common_utils


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * 4,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_embedding = nn.Parameter(torch.randn(1, 10, input_dim))  # 假设最大序列长度为10

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
        self.temporal_fusion_module = TemporalTransformer(
            input_dim=bev_feature_dim, num_heads=4, num_layers=1
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
            # # --- "诊断探针" 开始 ---
            # print("\n" + "=" * 60)
            # print(f"DEBUG: Processing Frame {t + 1}/{sequence_length}")
            # print("=" * 60)
            # 为每一帧创建一个独立的、包含了所有父级信息的batch_dict
            # 我们直接复用batch_dict，只替换掉与帧相关的数据
            frame_batch_dict = batch_dict.copy()
            for key, val in batch_dict.items():
                if isinstance(val, list) and len(val) == sequence_length:
                    frame_batch_dict[key] = val[t]

            common_utils.load_data_to_gpu(frame_batch_dict)
            # print(f"DEBUG (Before VFE): Keys in frame_batch_dict: {list(frame_batch_dict.keys())}")

            # 3. 让单帧数据流过VFE和3D骨干网络 (完全复用现有模块)
            # 这两个模块会直接修改 frame_batch_dict
            frame_batch_dict = self.vfe(frame_batch_dict)
            # print(f"DEBUG (After VFE): Keys in frame_batch_dict: {list(frame_batch_dict.keys())}")

            frame_batch_dict = self.backbone_3d(frame_batch_dict)

            # print(f"DEBUG (After Backbone_3D): Keys in frame_batch_dict: {list(frame_batch_dict.keys())}")
            # 4. 收集每一帧的BEV特征图
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
        B, T, C, H, W = bev_features_sequence.shape

        # 6. 为Transformer重塑(reshape)数据并进行融合
        bev_features_flat = bev_features_sequence.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        # 关键修正：分块处理以避免CUDA错误
        # ---
        fused_features_flat_list = []
        chunk_size = 1024 # 可以根据您的显存调整这个值
        for i in range(0, bev_features_flat.shape[0], chunk_size):
            chunk = bev_features_flat[i:i + chunk_size]
            fused_chunk = self.temporal_fusion_module(chunk)
            fused_features_flat_list.append(fused_chunk)

        fused_features_flat = torch.cat(fused_features_flat_list, dim=0)
        # ------------------------------------
        # 7. 送入Transformer并得到融合后的特征图
        # fused_features_flat = self.temporal_fusion_module(bev_features_flat)

        # 7. 只取最后一帧用于预测
        target_frame_features_flat = fused_features_flat[:, -1, :]

        # 8. 恢复BEV形状并送入预测头
        fused_bev_feature = target_frame_features_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 9. 准备最终送入预测头的数据字典
        final_batch_dict = {key: val[-1] for key, val in batch_dict.items() if isinstance(val, list)}
        final_batch_dict['batch_size'] = batch_size
        final_batch_dict['spatial_features_2d'] = fused_bev_feature
        # 确保最后一帧的GT等数据也上GPU


        final_batch_dict = self.dense_head(final_batch_dict)
        # ----------------------------------------------------

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(final_batch_dict)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            # post_processing期望batch_dict包含原始信息，我们从最后一帧获取
            # 并将dense_head的预测结果更新进去
            batch_dict_for_post = {key: val[-1] for key, val in batch_dict.items() if isinstance(val, list)}
            batch_dict_for_post.update(self.dense_head.forward_ret_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict_for_post)
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
