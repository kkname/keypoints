# 特征诊断系统使用说明

## 目的

帮助你理解Transformer融合特征后，数据在模型中的传递过程，找出为什么预测全空的根本原因。

## 诊断点位置

我在代码中插入了5个关键诊断点，它们会在**evaluation时**（不是training时）自动记录特征统计信息：

```
数据流：点云 → VFE → Backbone3D → BEV特征 → Transformer → Adapter → Sparse重建 → Dense Head → 预测

诊断点1: Transformer输出（adapter之前）
  位置: voxelnext_kp.py:214
  记录: transformer_output_before_adapter
  含义: Transformer刚输出的BEV特征，还没经过adapter

诊断点2: Adapter输出
  位置: voxelnext_kp.py:221
  记录: adapter_output
  含义: 经过特征适配层后的BEV特征

诊断点3: 重建的Sparse Tensor
  位置: voxelnext_kp.py:249
  记录: fused_sparse_tensor_to_densehead
  含义: 从dense BEV特征重建的sparse tensor，即将送入dense_head

诊断点4: Dense Head输入
  位置: voxelnext_head_kp_merge.py:241
  记录: densehead_input
  含义: dense_head实际接收到的输入

诊断点5: HM分支输出
  位置: voxelnext_head_kp_merge.py:250-260
  记录: head0_hm_logits, head0_hm_after_sigmoid
  含义:
    - hm_logits: sigmoid之前的原始值（这个决定最终检测）
    - hm_after_sigmoid: sigmoid之后的概率值
```

## 运行诊断

### 方法1：使用诊断脚本（推荐）

```bash
cd /root/autodl-tmp/VoxelKP

# 诊断最新的checkpoint
python tools/diagnose_features.py \
  --cfg_file tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml \
  --ckpt output/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel/pretrain_3frames_with_adapter/ckpt/checkpoint_epoch_1.pth \
  --num_samples 3

# 参数说明：
# --num_samples 3: 只诊断前3个样本（避免太慢）
```

### 方法2：在正常evaluation时查看

```bash
# 正常运行evaluation，诊断会自动打印到终端
python tools/test.py \
  --cfg_file tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml \
  --ckpt output/.../checkpoint_epoch_X.pth
```

## 诊断报告解读

运行后，你会看到类似这样的输出：

```
================================================================================
样本 1 特征诊断报告
================================================================================

[1] transformer_output_before_adapter
  Shape: (1, 384, 468, 468)
  Range: [-2.5431, 3.2145]       ← 数值范围
  Mean: 0.0234, Std: 0.8912      ← 均值和标准差
  Abs Mean: 0.6745               ← 绝对值均值
  Channel mean range: [-0.8234, 1.2345]  ← 不同通道的均值范围
  Channel std: 0.4567            ← 通道间的方差

[2] adapter_output
  Shape: (1, 384, 468, 468)
  Range: [-2.1234, 2.9876]
  Mean: 0.0189, Std: 0.7654
  ...

[3] fused_sparse_tensor_to_densehead
  Type: SparseConvTensor
  Active voxels: 45678, Channels: 384  ← 稀疏张量的非零体素数量
  Range: [-2.1234, 2.9876]
  ...

[4] densehead_input
  Type: SparseConvTensor
  Active voxels: 45678, Channels: 384
  Range: [-2.1234, 2.9876]
  ...

[5] head0_hm_logits
  Shape: (45678,)  ← 只有1维，每个体素一个值
  Range: [-73.7979, -0.9757]  ← ⚠️ 全是负值！问题所在！
  Mean: -35.2145, Std: 15.6789
  ...

[6] head0_hm_after_sigmoid
  Shape: (45678,)
  Range: [0.0000, 0.3769]  ← sigmoid后，全部 < 0.3阈值
  Mean: 0.0001, Std: 0.0045  ← 几乎全是0
  ...
```

## 关键分析指标

### 1. 对比Transformer前后的分布变化

**正常情况**（bypass模式）：
```
bypass模式的BEV特征:
  Range: [-165.18, 5.37]  ← 包含正值
  Mean: -15, Std: 20
```

**异常情况**（Transformer模式）：
```
transformer_output:
  Range: [-5.0, 2.0]  ← 范围变小了
  Mean: -1.5, Std: 0.8  ← 方差也变小了
```

👉 **说明Transformer平滑了特征**，把原本的"尖锐峰值"拉平了。

### 2. 检查Adapter是否起作用

对比 `transformer_output_before_adapter` 和 `adapter_output`:

**Adapter不起作用**（初始化问题）：
```
before: Range [-2.5, 3.2], Mean 0.02
after:  Range [-2.5, 3.2], Mean 0.02  ← 完全一样！
```

**Adapter起作用**：
```
before: Range [-2.5, 3.2], Mean 0.02, Std 0.89
after:  Range [-180, 6.5], Mean -12, Std 25  ← 分布被拉伸了！
```

👉 如果adapter_output和transformer_output完全一样，说明adapter的单位矩阵初始化没生效。

### 3. 检查HM分支的输出

**正常情况**（有检测）：
```
hm_logits:
  Range: [-165.18, 5.37]  ← 有正值！
  有少量位置 > 0，大部分 < 0

hm_after_sigmoid:
  Range: [0.0000, 0.9952]  ← 最大值接近1
  有部分位置 > 0.3阈值
```

**异常情况**（无检测）：
```
hm_logits:
  Range: [-73.80, -0.98]  ← 全是负值！

hm_after_sigmoid:
  Range: [0.0000, 0.2727]  ← 最大值 = sigmoid(-0.98) = 0.27 < 0.3
  全部位置 < 0.3阈值 → 0个检测
```

## 诊断方案

基于诊断报告，按这个流程分析：

### 步骤1：对比transformer_output vs bypass

**如果你有bypass模式的checkpoint**：
- 运行bypass模式，记录特征分布
- 运行transformer模式，对比分布差异
- **差异很大** → Transformer改变了特征语义
- **差异很小** → 说明Transformer学习失败（可能欠拟合）

### 步骤2：检查Adapter效果

对比`transformer_output_before_adapter` vs `adapter_output`:

```python
# 计算差异
mean_diff = |before.mean - after.mean|
std_diff = |before.std - after.std|

if mean_diff < 0.01 and std_diff < 0.01:
    print("⚠️ Adapter没起作用！检查初始化")
else:
    print("✓ Adapter在学习")
```

### 步骤3：追踪到HM分支

检查`hm_logits`的range:

```python
if hm_logits.max() < 0:
    print("❌ 问题确诊：HM全是负值")
    print("原因可能是：")
    print("  1. Transformer输出的特征分布不对")
    print("  2. Adapter没能恢复正确的分布")
    print("  3. Dense head的卷积权重不匹配新特征")
```

### 步骤4：定位问题环节

对比各诊断点，找出"特征崩溃"的位置：

```
诊断点1: Range [-2.5, 3.2]   ← OK
诊断点2: Range [-2.5, 3.2]   ← OK（但可能adapter没起作用）
诊断点3: Range [-2.5, 3.2]   ← OK
诊断点4: Range [-2.5, 3.2]   ← OK
诊断点5: Range [-73, -0.9]   ← ❌ 崩溃点！

→ 问题出在dense_head的HM卷积层内部！
→ 输入看起来正常，但经过卷积后变成全负值
→ 说明：卷积权重与新特征分布不匹配
```

## 下一步行动

根据诊断结果，可能的解决方案：

### 情况A：Adapter没起作用
```
症状：adapter_output = transformer_output（完全一样）

解决方案：
1. 检查adapter的初始化代码
2. 确认adapter在optimizer中（是否被freeze了）
3. 检查学习率是否太小
```

### 情况B：Adapter起作用但不够
```
症状：adapter_output略有变化，但hm_logits仍然全负

解决方案：
1. 增大adapter的学习率
2. 使用更强的adapter（如2层1×1卷积）
3. 考虑添加残差连接
```

### 情况C：根本性不兼容
```
症状：即使adapter学习很好，hm依然全负

解决方案：
1. 微调dense_head（不freeze）
2. 改用简单的时序融合（加权平均）
3. 重新设计temporal fusion架构
```

## 快速测试命令

```bash
# 1. 诊断第1轮checkpoint（初始状态）
python tools/diagnose_features.py \
  --cfg_file tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml \
  --ckpt output/.../checkpoint_epoch_1.pth \
  --num_samples 1

# 2. 对比bypass模式（如果有）
python tools/diagnose_features.py \
  --cfg_file tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml \
  --ckpt best_model/checkpoint_epoch_500.pth \  # 单帧模型
  --num_samples 1

# 3. 查看训练中的checkpoint
python tools/diagnose_features.py \
  --cfg_file tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml \
  --ckpt output/.../checkpoint_epoch_5.pth \
  --num_samples 1
```

## 注意事项

1. **诊断只在evaluation模式运行**，training时不会记录（避免影响性能）
2. **每个样本都会打印一次报告**，如果觉得太多，减少`--num_samples`
3. **诊断会略微降低推理速度**（约10-20%），因为要统计特征
4. **报告中的数值是单个样本的统计**，不同样本可能有差异

## 我帮你分析

运行诊断后，把输出发给我，我会帮你：
1. 解读每个诊断点的含义
2. 找出特征崩溃的具体位置
3. 建议针对性的解决方案
4. 判断是否需要调整架构

开始吧！先运行第一个诊断命令，看看结果。
