# Plan B: 残差连接方案

## 问题
即使adapter初始化正确，Transformer的特征可能与预训练dense_head根本不兼容。

## 解决方案：保留原始特征 + 添加时序增强

### 核心思想
不要完全替换最后一帧的特征，而是在原始特征基础上**添加**Transformer的增强信息。

### 代码修改

在 voxelnext_kp.py 的 forward 函数中：

```python
# 当前代码（第204-217行）
# 重塑回BEV格式: (B, H*W, C) -> (B, C, H, W)
fused_bev_feature = fused_features_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

# 清理中间变量
del bev_features_sequence, bev_features_flat, fused_features_chunks, fused_features_flat
torch.cuda.empty_cache()

# 7. 应用特征适配层，对齐特征分布
fused_bev_feature = self.feature_adapter(fused_bev_feature)
```

**改为：**

```python
# 重塑回BEV格式: (B, H*W, C) -> (B, C, H, W)
temporal_enhanced_features = fused_features_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

# 获取当前帧的原始BEV特征（最后一帧）
current_frame_features = bev_features_sequence[:, -1, :, :, :]  # (B, C, H, W)

# 清理中间变量
del bev_features_sequence, bev_features_flat, fused_features_chunks, fused_features_flat
torch.cuda.empty_cache()

# 7. 残差连接：原始特征为主 + 时序增强为辅
# 通过adapter学习合适的混合比例
temporal_delta = self.feature_adapter(temporal_enhanced_features - current_frame_features)
fused_bev_feature = current_frame_features + 0.1 * temporal_delta
```

### 为什么这样更好？

1. **保留原始特征**：
   - 预训练的dense_head能识别 `current_frame_features`
   - 确保基础检测能力

2. **渐进增强**：
   - `temporal_delta` 是Transformer学到的"时序差异"
   - 系数 `0.1` 让增强缓慢生效

3. **可学习融合**：
   - Adapter不是学习"完全变换"
   - 而是学习"需要多少时序信息"

### 预期效果

- **训练初期**：adapter权重≈0，模型≈bypass模式（recall ≈ 0.5）
- **训练中期**：adapter学会提取有用的时序信号
- **训练后期**：性能超越单帧baseline

### 使用方法

1. 修改 voxelnext_kp.py 第204-217行
2. 从预训练模型重新开始训练
3. 观察recall是否从epoch 1就>0（说明残差连接work）
