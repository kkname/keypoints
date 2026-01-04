# VoxelKP 稀疏时序融合架构 - 逐层数据流分析

**架构版本**: Sparse Temporal Fusion v1.0
**日期**: 2026-01-04

完整解析从点云输入到最终3D检测结果的每一层数据变换（基于新的稀疏架构）。

---

## 🔄 完整的逐层数据流

### Layer 0: 原始输入 - 点云

**输入数据**：
```python
points: (N_points, 5)
# 每个点: [x, y, z, intensity, timestamp]
# x,y,z: 3D坐标（米）
# intensity: 激光反射强度
# timestamp: 相对于当前帧的时间偏移
```

**时序输入（3帧序列）**：
```python
batch_dict = {
    'points': [points_t0, points_t1, points_t2],  # 3帧点云
    'frame_id': ['frame_1000', 'frame_1001', 'frame_1002'],
    'gt_boxes': [boxes_t0, boxes_t1, boxes_t2],  # (B, M, 8)
    'keypoint_location': [kps_t0, kps_t1, kps_t2],  # (B, M, 14, 3)
    'batch_size': 2  # 每帧2个样本
}
```

**示例**：
```python
points_t2 = [  # 当前帧
    [2.5, 1.3, 0.8, 0.45, 0.0],   # 点1
    [2.6, 1.4, 0.9, 0.52, 0.0],   # 点2
    ...
    [-3.2, 5.1, 1.2, 0.31, 0.0]   # 点N
]
```

---

## Layer 1: VFE (Voxel Feature Encoding) - 逐帧处理

### 输入（每帧独立）
```python
# 对每一帧 t ∈ {0, 1, 2}:
points_t: (N_points_t, 5)
```

### 体素化
```python
空间范围：x∈[-75.2, 75.2], y∈[-75.2, 75.2], z∈[-2, 4]
体素大小：0.1m × 0.1m × 0.15m
体素网格：1504 × 1504 × 40 ≈ 90M个格子

实际有点的体素：~3500个/帧（稀疏度：99.996%）
```

### VFE处理（MeanVFE）
```python
# 对每个体素内的点取平均
voxel_feature = mean(points_in_voxel, dim=points_axis)

输出（每帧）：
voxel_features: (num_voxels, 5)
voxel_coords: (num_voxels, 4)  # [batch_idx, z, y, x]
num_points_per_voxel: (num_voxels,)
```

**示例**：
```python
# 帧t的输出
voxel_coords_t = [
    [0, 12, 750, 800],  # batch 0, z=12层, y=750, x=800
    [0, 12, 751, 800],
    [0, 13, 750, 800],
    ...
]  # (3500, 4)

voxel_features_t: (3500, 5)
```

---

## Layer 2: Backbone 3D - Sparse 3D Convolutions（逐帧处理）

### 输入（每帧）
```python
SparseConvTensor(
    features: (3500, 5),         # 初始特征
    indices: (3500, 4),          # [batch, z, y, x]
    spatial_shape: [40, 1504, 1504],
    batch_size: 2
)
```

### Backbone结构
```python
# 4个稀疏卷积block + BEV压缩
block1: 5 channels  → 16 channels   (stride=1)
  └─ output: (3500, 16), spatial: [40, 1504, 1504]

block2: 16 channels → 32 channels   (stride=2, 空间减半)
  └─ output: (2800, 32), spatial: [20, 752, 752]

block3: 32 channels → 64 channels   (stride=2)
  └─ output: (2100, 64), spatial: [10, 376, 376]

block4: 64 channels → 128 channels  (stride=2)
  └─ output: (1500, 128), spatial: [5, 188, 188]

# BEV压缩：将Z维度压缩到1层
bev_conv: 128 → 384 channels
  └─ output: (1200, 384), spatial: [1, 188, 188]
```

### 最终输出（每帧）
```python
encoded_spconv_tensor_t:
    features: (1200, 384),  # BEV前景点
    indices: (1200, 4),     # [batch, z=0, y, x]
    spatial_shape: [1, 188, 188],  # Z=1（BEV）
    batch_size: 2

特征语义：
- 384维高级特征
- 编码了周围数米范围的几何模式
- 包含物体形状、姿态的抽象信息
```

### 收集3帧
```python
# voxelnext_kp.py: 第123-128行
bev_features_list = [
    sparse_tensor_t0,  # (1200, 384), spatial=[1, 188, 188]
    sparse_tensor_t1,  # (1100, 384), spatial=[1, 188, 188]
    sparse_tensor_t2,  # (1250, 384), spatial=[1, 188, 188]
]
# 注意：每帧的非零体素数量可能不同！
```

---

## Layer 3: Sparse Temporal Fusion - 核心创新

### 输入
```python
sparse_bev_sequence: List[SparseConvTensor]  # 长度=3
# 每个元素：
#   features: (N_t, 384)
#   indices: (N_t, 4) [batch, z, y, x]
#   spatial_shape: [1, 188, 188]
```

---

### 3.1 稀疏化 + 时间编码

**目的**：将3帧的稀疏点合并，并添加时间标记

**过程**：
```python
all_features = []
all_indices = []
point_frame_ids = []

for t, sparse_tensor in enumerate(sparse_bev_sequence):  # t=0,1,2
    feats = sparse_tensor.features  # (N_t, 384)
    indices = sparse_tensor.indices  # (N_t, 4)

    # 添加时间编码：让网络知道这个点来自哪一帧
    time_emb = time_embedding(t)  # (384,) 可学习的向量
    feats_with_time = feats + time_emb

    all_features.append(feats_with_time)
    all_indices.append(indices)
    point_frame_ids.extend([t] * N_t)

# 拼接所有帧的点
flat_features = torch.cat(all_features, dim=0)  # (N_total, 384)
flat_indices = torch.cat(all_indices, dim=0)    # (N_total, 4)
```

**输出**：
```python
flat_features: (N_total, 384)  # N_total ≈ 1200+1100+1250 = 3550
flat_indices: (N_total, 4)
point_frame_ids: (N_total,)  # [0,0,...,1,1,...,2,2,...]

时间编码的作用：
- t=0的点：features + time_emb[0]  # "我是2帧前的历史"
- t=1的点：features + time_emb[1]  # "我是1帧前的历史"
- t=2的点：features + time_emb[2]  # "我是当前帧"
```

---

### 3.2 稀疏注意力（VarLengthMultiheadSA）

**输入**：
```python
sptr_tensor = SparseTrTensor(
    query_feats: (3550, 384),
    query_indices: (3550, 4),
    spatial_shape: [1, 188, 188],
    batch_size: 2
)
```

**窗口划分**：
```python
window_size = [10, 10, 1]  # BEV平面10×10窗口，Z维度=1

空间分辨率：
- 每个BEV像素 = 0.4m × 0.4m（因为stride=2了3次）
- 10个像素 = 4m × 4m

窗口内点的聚合：
- 同一窗口内的点（跨帧）会做注意力交互
- 例如：位置(100, 120)在10×10窗口[10, 12]中
  - t=0的点：(0, 0, 101, 118) → 窗口[10, 11]
  - t=1的点：(0, 0, 102, 121) → 窗口[10, 12]
  - t=2的点：(0, 0, 100, 120) → 窗口[10, 12]
  这3个点会在窗口[10, 12]内做注意力！
```

**多头自注意力**：
```python
# 8个注意力头
num_heads = 8
head_dim = 384 // 8 = 48

# 对每个窗口内的点
Q = linear_q(query_feats)  # (N_window, 8, 48)
K = linear_k(query_feats)  # (N_window, 8, 48)
V = linear_v(query_feats)  # (N_window, 8, 48)

# 注意力权重
Attention = softmax(Q @ K^T / sqrt(48))  # (N_window, N_window)

# 窗口内注意力示例（假设窗口有5个点：2个t=0，1个t=1，2个t=2）
Attention_example = [
    # Q\K  t0_p1 t0_p2 t1_p1 t2_p1 t2_p2
    [0.3,  0.2,  0.1,  0.2,  0.2],  # t0_p1看谁
    [0.2,  0.3,  0.1,  0.2,  0.2],  # t0_p2看谁
    [0.1,  0.1,  0.4,  0.2,  0.2],  # t1_p1主要看自己
    [0.15, 0.15, 0.2,  0.3,  0.2],  # t2_p1看当前+历史
    [0.1,  0.1,  0.1,  0.2,  0.5],  # t2_p2主要看自己
]

# 加权求和
Output = Attention @ V  # (N_window, 8, 48)
```

**物理含义**：
```
当前帧的点可以"看到"历史帧的点：
- 如果当前帧某位置被遮挡 → 注意力权重高的是历史帧的点
- 如果当前帧清晰可见 → 注意力权重高的是自己

例如：行人走过遮挡物
- t=0: 人在遮挡物左边（清晰）
- t=1: 人在遮挡物后面（部分遮挡）
- t=2: 人在遮挡物右边（清晰）

t=1的遮挡位置会"借用"t=0和t=2的信息来修复！
```

**输出**：
```python
output_tensor.query_feats: (3550, 384)
# 每个点的特征现在融合了窗口内其他点（跨帧）的信息
```

---

### 3.3 LayerNorm + FFN

**LayerNorm（安全）**：
```python
# 只归一化前景点（N_total个），不影响背景
normalized = LayerNorm(output_feats + flat_features)  # 残差连接

特征分布：
输入（时间编码后）: mean ≈ -15 (保留原始分布)
输出（LayerNorm后）: mean ≈ 0, std ≈ 1

关键：背景（0值）不在这个List里，完全不受影响！
```

**FFN（前馈网络）**：
```python
# 增强特征表达能力
ffn_out = ReLU(Linear1(normalized))  # (N_total, 768) 扩展2倍
final = Linear2(ffn_out)             # (N_total, 384)

# 残差连接
fused_features = normalized + final
```

---

### 3.4 提取当前帧

**只保留当前帧（t=2）的点**：
```python
current_frame_mask = (point_frame_ids == 2)

current_features = fused_features[current_frame_mask]  # (N_current, 384)
current_indices = flat_indices[current_frame_mask]      # (N_current, 4)

# 重建当前帧的SparseConvTensor
fused_sparse_bev = SparseConvTensor(
    features: (N_current, 384),  # 约1250个点
    indices: (N_current, 4),
    spatial_shape: [1, 188, 188],
    batch_size: 2
)
```

**关键点**：
- 历史帧的点（t=0, t=1）被丢弃，只用于注意力计算
- 当前帧的点经过时序融合，包含了历史信息

---

## Layer 4: Adapter - 分布对齐

### 4.1 Sparse → Dense 转换

**输入**：
```python
fused_sparse_bev:
    features: (1250, 384)
    indices: (1250, 4) [batch, z, y, x]
    spatial_shape: [1, 188, 188]
```

**转Dense**：
```python
fused_dense = fused_sparse_bev.dense()  # (B, C, Z, H, W)
                                        # (2, 384, 1, 188, 188)

# 压缩Z维度
assert fused_dense.shape[2] == 1
fused_dense = fused_dense.squeeze(2)   # (2, 384, 188, 188)
```

---

### 4.2 BatchNorm2d 对齐

**Adapter结构**：
```python
adapter = nn.Sequential(
    nn.Conv2d(384, 384, 1, bias=False),  # 恒等变换
    nn.BatchNorm2d(384, affine=True)     # 分布对齐
)
```

**Conv2d（恒等初始化）**：
```python
# weight初始化为单位矩阵
weight = eye(384).unsqueeze(-1).unsqueeze(-1)  # (384, 384, 1, 1)

输出 = Conv(输入) = 输入（无变换）
```

**BatchNorm2d（自动学习）**：
```python
# 归一化 + 可学习的缩放和平移
BN(x) = γ * (x - μ_batch) / σ_batch + β

初始化：
γ = 1.0  (scale)
β = 0.0  (shift)

训练后（自动学习）：
假设Transformer输出 mean≈0, std≈1
Dense Head期望 mean≈-15, std≈20

网络会学到：
γ ≈ 20  (放大方差)
β ≈ -15 (平移均值)

→ 输出分布自动对齐到Dense Head期望的分布！
```

**输出**：
```python
aligned_dense: (2, 384, 188, 188)

分布：
- 初始（bypass）: mean ≈ 0, std ≈ 1
- 训练后（学习）: mean ≈ -15, std ≈ 20
```

---

### 4.3 Dense → Sparse 重建

**提取稀疏位置的特征**：
```python
indices = fused_sparse_bev.indices  # (1250, 4) [batch, z, y, x]

batch_idx = indices[:, 0].long()  # (1250,)
y_idx = indices[:, 2].long()      # (1250,)
x_idx = indices[:, 3].long()      # (1250,)

# 从aligned_dense中提取
aligned_features = aligned_dense[batch_idx, :, y_idx, x_idx]  # (1250, 384)

# 重建SparseConvTensor
aligned_sparse = SparseConvTensor(
    features: (1250, 384),
    indices: (1250, 4),
    spatial_shape: [1, 188, 188],
    batch_size: 2
)
```

**关键**：
- 前景点的特征经过BN对齐
- 背景保持0（不在indices里，没有被处理）

---

## Layer 5: Dense Head - 多分支预测

### 输入
```python
x_2d = aligned_sparse:
    features: (1250, 384)
    indices: (1250, 4)
    spatial_shape: [1, 188, 188]
```

---

### 5.1 HM分支（Heatmap）

**结构**：
```python
SubMConv2d(384, 384, kernel=3×3) + BN + ReLU
SubMConv2d(384, 1, kernel=1×1)
```

**输出**：
```python
hm_logits: (1250,)  # 每个前景体素的物体中心概率（未归一化）

理想分布（训练收敛后）：
- 有物体的位置：> 0 (sigmoid后 > 0.5)
- 背景位置：< -2 (sigmoid后 < 0.12)

数值示例：
[5.2, 3.8, -0.5, -1.2, ..., -50.3, -120.5]
 ↑有物体  ↑边界    ↑不确定     ↑背景
```

---

### 5.2 其他分支

**位置分支（loc_x, loc_y, loc_z）**：
```python
输出: (1250, 15)  # 1个中心 + 14个关键点

loc_x[:, 0]: 物体中心的x偏移（相对体素中心）
loc_x[:, 1:15]: 14个关键点的x坐标（相对物体中心）

同理 loc_y, loc_z
```

**尺寸分支（dim）**：
```python
输出: (1250, 3)
含义: [log(长), log(宽), log(高)]  # 对数空间
实际尺寸 = exp(dim)
```

**旋转分支（rot）**：
```python
输出: (1250, 2)
含义: [cos(θ), sin(θ)]
角度 = atan2(sin, cos)
```

**可见性分支（kp_vis）**：
```python
输出: (1250, 14)
含义: 14个关键点的可见性logits
可见性概率 = sigmoid(kp_vis)
```

**汇总**：
```python
pred_dict = {
    'hm': (1250,),        # Heatmap logits
    'loc_x': (1250, 15),  # X方向位置
    'loc_y': (1250, 15),  # Y方向位置
    'loc_z': (1250, 15),  # Z方向位置
    'dim': (1250, 3),     # 尺寸（对数）
    'rot': (1250, 2),     # 旋转（sin/cos）
    'kp_vis': (1250, 14), # 关键点可见性
}
```

---

## Layer 6: Sigmoid & TopK选择

### HM Sigmoid
```python
hm_probs = sigmoid(hm_logits)  # (1250,)

理想情况：
[0.995, 0.978, 0.377, 0.231, ..., 0.001, 0.000]
 ↑有物体            ↑边界          ↑背景
```

### TopK选择
```python
K = 500  # MAX_OBJ_PER_SAMPLE

scores, inds = torch.topk(hm_probs, K)
# 选择概率最高的500个位置
```

### 阈值过滤
```python
SCORE_THRESH = 0.3

mask = scores > 0.3
valid_inds = inds[mask]  # 可能0到500个

# 提取有效预测
final_hm = hm_probs[valid_inds]
final_loc_x = loc_x[valid_inds]
...
```

---

## Layer 7: 3D框 + 关键点解码

### 解码过程
```python
for each valid index i:
    # 体素中心坐标（世界坐标系）
    voxel_center = (
        indices[i, 3] * 0.4 + (-75.2),  # x
        indices[i, 2] * 0.4 + (-75.2),  # y
        indices[i, 1] * 0.15 + (-2.0)   # z
    )

    # 物体中心 = 体素中心 + 预测偏移
    center_x = voxel_center_x + loc_x[i, 0] * 0.4
    center_y = voxel_center_y + loc_y[i, 0] * 0.4
    center_z = loc_z[i, 0]

    # 尺寸
    l, w, h = exp(dim[i])

    # 角度
    angle = atan2(rot[i, 1], rot[i, 0])

    # 3D框
    box_3d = [center_x, center_y, center_z, l, w, h, angle]

    # 关键点
    for j in range(14):
        kp_x = center_x + loc_x[i, j+1] * 0.4
        kp_y = center_y + loc_y[i, j+1] * 0.4
        kp_z = loc_z[i, j+1]
        kp_vis = sigmoid(kp_vis[i, j])

        keypoints[j] = [kp_x, kp_y, kp_z, kp_vis]
```

### 最终输出
```python
final_pred_dict = {
    'pred_boxes': (N_detected, 7),     # [x, y, z, l, w, h, θ]
    'pred_kps': (N_detected, 14, 3),   # 14个关键点3D坐标
    'pred_kps_vis': (N_detected, 14),  # 可见性
    'pred_scores': (N_detected,),      # 置信度
    'pred_labels': (N_detected,),      # 类别（人=1）
}

示例：N_detected = 15（检测到15个人）
```

---

## 📊 完整流程总结表

| Layer | 输入 Shape | 输出 Shape | 数据类型 | 关键操作 |
|-------|-----------|-----------|---------|---------|
| **0. 点云** | `(N, 5)` × 3帧 | - | List[Tensor] | 原始输入 |
| **1. VFE** | `(N, 5)` × 3 | `(N_v, 5)` × 3 | List[Tensor] | 体素化（逐帧） |
| **2. Backbone3D** | `(N_v, 5)` × 3 | `(~1200, 384)` × 3 | List[Sparse] | 3D卷积（逐帧） |
| **3.1 稀疏化** | List[Sparse] × 3 | `(3550, 384)` | Tensor | 合并+时间编码 |
| **3.2 稀疏注意力** | `(3550, 384)` | `(3550, 384)` | SparseTensor | 窗口注意力 |
| **3.3 LayerNorm+FFN** | `(3550, 384)` | `(3550, 384)` | Tensor | 归一化+增强 |
| **3.4 提取当前帧** | `(3550, 384)` | `(1250, 384)` | Sparse | 过滤t=2 |
| **4.1 Sparse→Dense** | Sparse `(1250, 384)` | `(2, 384, 188, 188)` | Dense | 格式转换 |
| **4.2 Adapter** | `(B, 384, H, W)` | `(B, 384, H, W)` | Dense | Conv+BN对齐 |
| **4.3 Dense→Sparse** | Dense `(B, 384, H, W)` | Sparse `(1250, 384)` | Sparse | 提取前景 |
| **5. Dense Head** | Sparse `(1250, 384)` | `pred_dict` | Dict | 多分支预测 |
| **6. Sigmoid+TopK** | `hm: (1250,)` | `scores: (500,)` | Tensor | 候选选择 |
| **7. 解码** | pred_dict | `(N, 7), (N, 14, 3)` | Tensor | 3D框+关键点 |

---

## 🔑 关键创新点

### 1. 全程保持稀疏性
```
旧架构：Sparse → Dense (BEV) → Transformer → Dense Head
新架构：Sparse → Sparse Fusion → (临时Dense) → Sparse → Dense Head

内存占用：
旧: ~4GB（Dense BEV: 2×3×384×188×188）
新: ~0.5GB（Sparse: 3550×384）
节省: 87%
```

### 2. LayerNorm不破坏背景
```
旧架构（Dense）:
- LayerNorm归一化全图（包括背景0）
- 背景0 → 被拉到均值附近（-0.005）
- 破坏了稀疏假设

新架构（Sparse）:
- LayerNorm只归一化前景点（3550个）
- 背景根本不在List里，保持纯0
- 完美保持稀疏性！
```

### 3. 时间编码
```
让网络明确知道每个点来自哪一帧：
- t=0的点：time_emb[0]  # "我是历史"
- t=2的点：time_emb[2]  # "我是当前"

网络可以学会：
- 当前帧被遮挡 → 高权重给历史帧
- 当前帧清晰 → 高权重给自己
```

### 4. 窗口注意力
```
只在空间邻近的点之间做注意力：
- 窗口大小：10×10 ≈ 4m×4m
- 符合物理直觉（远处的点不相关）
- 大幅降低计算量
```

### 5. BatchNorm自动对齐
```
旧架构：手工初始化Adapter为恒等变换（难调）
新架构：BatchNorm自动学习分布映射

BN自动学习：
γ ≈ 20  (scale)
β ≈ -15 (shift)

→ Transformer输出(mean=0) 自动映射到 Dense Head期望(mean=-15)
```

---

## 🎯 预期效果

### Epoch 1（初始化正确）
```
训练Loss：正常下降（不是NaN）
Recall@0.3：> 0.3（说明BatchNorm对齐成功）
Recall@0.5：> 0.1
```

### Epoch 20+（收敛后）
```
Recall@0.3：0.65 - 0.70
Recall@0.5：0.50 - 0.55
Recall@0.7：0.35 - 0.40

优于单帧baseline（0.60 / 0.45 / 0.30）
```

---

## 📁 代码文件索引

| 文件 | 行号 | 功能 |
|------|------|------|
| `voxelnext_kp.py` | 32-38 | Sparse Temporal Fusion初始化 |
| `voxelnext_kp.py` | 119-128 | 逐帧Backbone处理 + 收集 |
| `voxelnext_kp.py` | 133 | 稀疏时序融合调用 |
| `sparse_temporal_fusion.py` | 59-126 | 稀疏融合模块 |
| `sparse_temporal_fusion.py` | 153-245 | Adapter + 重建 |
| `voxelnext_head_kp_merge.py` | 222-229 | Dense Head统一Sparse输入 |
| `voxelnext_head_kp_merge.py` | 237-265 | 多分支预测 |

---

## 💡 与旧架构对比

| 维度 | 旧架构（Dense Transformer） | 新架构（Sparse Fusion） |
|------|---------------------------|------------------------|
| **数据格式** | Sparse→Dense→Sparse | 全程Sparse（只Adapter临时Dense） |
| **LayerNorm** | 归一化全图（破坏背景） | 只归一化前景点 |
| **时间编码** | 位置编码（隐式） | 显式Time Embedding |
| **注意力** | 全局（H×W个位置） | 窗口内（10×10邻域） |
| **分布对齐** | 手工初始化Conv | BatchNorm自动学习 |
| **内存占用** | ~4GB | ~0.5GB（↓87%） |
| **计算效率** | 处理219k像素 | 处理3.5k前景点（↓98%） |
| **背景保真** | 被LayerNorm破坏 | 完美保持0 |

---

**架构升级完成！准备训练！** 🚀
