# VoxelKP 逐层数据流分析

完整解析从点云输入到最终3D检测结果的每一层数据变换。

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

**示例**：
```python
[[2.5, 1.3, 0.8, 0.45, 0.0],   # 点1
 [2.6, 1.4, 0.9, 0.52, 0.0],   # 点2
 ...
 [-3.2, 5.1, 1.2, 0.31, 0.0]]  # 点N
```

---

## Layer 1: VFE (Voxel Feature Encoding)

### 作用
把点云体素化，提取每个体素的特征

### 输出
```python
voxel_features: (num_voxels, max_points_per_voxel, 5)
voxel_coords: (num_voxels, 4)  # [batch_idx, z, y, x]
num_points_per_voxel: (num_voxels,)
```

### 含义
- 把3D空间划分成小格子（体素）
- 每个体素大小：0.1m × 0.1m × 0.15m
- 把落在同一体素内的点聚合

### 体素化示例
```
空间范围：x∈[-75.2, 75.2], y∈[-75.2, 75.2], z∈[-2, 4]
体素数量：约 1504 × 1504 × 40 ≈ 90M个格子（但大部分是空的）

实际有点的体素：~3500个（稀疏！）

voxel_coords示例：
[[0, 12, 750, 800],  # batch 0, z=12层, y=750列, x=800行
 [0, 12, 751, 800],
 [0, 13, 750, 800],
 ...]
```

### VFE处理（MeanVFE）
```python
# 对每个体素内的点取平均
voxel_feature = mean(points_in_voxel, dim=points)

输出：voxel_features: (num_voxels, 5)
# 每个体素一个5维特征向量
```

---

## Layer 2: Backbone 3D - Sparse 3D Convolutions

### 输入
```python
SparseConvTensor(
    features: (num_voxels, 5),       # 初始特征
    indices: (num_voxels, 4),        # [batch, z, y, x]
    spatial_shape: [40, 1504, 1504], # Z, Y, X
    batch_size: 3
)
```

### Backbone结构（简化）
```python
# 4个稀疏卷积block
block1: 5 channels  → 16 channels   (stride=1)
  └─ output: (num_voxels_1, 16)

block2: 16 channels → 32 channels   (stride=2, 空间减半)
  └─ output: (num_voxels_2, 32)
  └─ spatial_shape: [20, 752, 752]

block3: 32 channels → 64 channels   (stride=2)
  └─ output: (num_voxels_3, 64)
  └─ spatial_shape: [10, 376, 376]

block4: 64 channels → 128 channels  (stride=2)
  └─ output: (num_voxels_4, 128)
  └─ spatial_shape: [5, 188, 188]
```

### 每个block做什么？

**Block 1输出示例**：
```python
SparseConvTensor(
    features: (3500, 16),  # 3500个非零体素，每个16维特征
    indices: (3500, 4),
    spatial_shape: [40, 1504, 1504]
)

特征含义：
- 16维向量编码了"这个体素周围的几何模式"
- 例如：
  - 维度0-3: 边缘检测（上下左右）
  - 维度4-7: 角点检测
  - 维度8-11: 表面法向量
  - 维度12-15: 密度信息
```

### 最后输出（block4后）
```python
encoded_spconv_tensor:
    features: (~1200, 128),  # 经过3次stride=2，体素数量减少
    indices: (1200, 4),
    spatial_shape: [5, 188, 188]  # Z维度被压缩到5层
```

### 特征语义
- 128维高级特征
- 编码了更大范围的上下文（感受野~数米）
- 包含物体形状、姿态的抽象信息

---

## Layer 3: BEV Feature Extraction

### 作用
从3D特征提取鸟瞰图（Bird's Eye View）特征

### 方法
```python
# voxelnext_kp.py: 152-154行
bev_feature_map = sparse_tensor.dense()
# 把稀疏3D张量转为密集张量，然后取某一层

输出：
bev_feature_map: (B, 128, 5, 188, 188)
# 注意：还有Z维度！
```

### 实际BEV输出
```python
bev_features: (B, 384, 188, 188)
# B=batch_size（这里是3帧的batch，所以B=3或6取决于配置）
# 384 = BEV特征通道数
# 188×188 = 空间分辨率

空间对应关系：
- 每个像素对应物理空间：0.4m × 0.4m (因为stride=2了3次)
- 覆盖范围：约 75m × 75m
```

### BEV特征的含义

每个位置 (i, j) 的384维向量编码：
```
bev_features[0, :, i, j] = 384维向量
含义：
- "在地面位置(i, j)处，从地面到2米高度范围内的3D信息"
- 包括：
  - 是否有物体
  - 物体的高度分布
  - 物体的密度
  - 物体的几何形状（在Z方向的投影）
```

### 可视化理解
```
想象从天上往下看：

    y
    ↑
    |  [人]  [车]
    |
    |        [树]
    |
    +----------→ x

BEV特征图就是这个俯视图，每个位置存储384维特征
```

---

## Layer 4: Temporal Transformer

### 输入（对于3帧序列）
```python
bev_features_sequence: (B, T, C, H, W)
                     = (1, 3, 384, 188, 188)
# B=1 (一个batch，但包含3帧数据)
# T=3 (时间维度：t-2, t-1, t)
# C=384 (通道)
# H=W=188 (空间)
```

### Transformer处理（分块处理）

```python
# 为了节省显存，对每个空间位置的时序特征分别处理

对于位置(i, j):
    输入: [frame_t-2[i,j], frame_t-1[i,j], frame_t[i,j]]
          shape: (3, 384)

    # Multi-Head Self-Attention
    Q = Linear_q(input)   # (3, 384)
    K = Linear_k(input)   # (3, 384)
    V = Linear_v(input)   # (3, 384)

    Attention = softmax(Q @ K^T / sqrt(384))  # (3, 3)
    # Attention矩阵示例：
    # [[0.4, 0.3, 0.3],   # t-2帧看t-2(0.4), t-1(0.3), t(0.3)
    #  [0.2, 0.5, 0.3],   # t-1帧看...
    #  [0.1, 0.3, 0.6]]   # t帧主要看自己(0.6)，但也看历史

    Output = Attention @ V  # (3, 384)

    # Feed-Forward Network
    FFN_out = ReLU(Linear1(Output))  # (3, 768) 中间扩展
    Final = Linear2(FFN_out)         # (3, 384)

    # 取最后一帧
    fused_feature[i,j] = Final[2]  # (384,)
```

### Transformer输出
```python
fused_features: (B, H*W, C) = (1, 35344, 384)
# 35344 = 188×188 个空间位置

重塑后：
fused_bev_feature: (B, C, H, W) = (1, 384, 188, 188)
```

### 关键效应（从诊断数据）
```
输入（单帧原始）：mean ≈ -15, std ≈ 20（理论值，bypass模式）
输出（Transformer）：mean ≈ -0.005, std ≈ 0.986

→ Transformer内部的LayerNorm归一化了特征！
```

### 物理含义

每个位置的特征不再是"瞬时状态"，而是"时序融合的状态"：
```
原始特征[i,j]：
  "t时刻，位置(i,j)有多大可能有物体"

融合特征[i,j]：
  "综合t-2, t-1, t三帧信息，位置(i,j)的状态"
  包含了：
  - 当前是否有物体
  - 物体的运动趋势
  - 时序一致性信息
```

---

## Layer 5: Feature Adapter

### 输入
```python
fused_bev_feature: (3, 384, 188, 188)
# 注意：这里B=3可能是因为batch包含3个样本
```

### Adapter结构
```python
# 1×1卷积
Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=True)

权重shape: (384, 384, 1, 1)
bias shape: (384,)
```

### 计算过程
```python
# 对每个位置(i,j)和每个batch
for b in range(B):
    for i in range(H):
        for j in range(W):
            input_vec = fused_bev_feature[b, :, i, j]  # (384,)

            # 1×1卷积 = 全连接（对通道维度）
            output_vec = weight @ input_vec + bias
            # weight: (384, 384)
            # input_vec: (384, 1)
            # output_vec: (384, 1)

            adapter_output[b, :, i, j] = output_vec
```

### 理想情况（单位矩阵初始化）
```python
weight = eye(384)  # 单位矩阵
bias = 0

输出 = 输入（恒等变换）
```

### 实际情况（诊断数据）
```
输入: mean = -0.005, std = 0.986
输出: mean = -1.287, std = 1.415

说明Adapter学到了变换：
output ≈ 1.435 × input - 1.28
```

### 输出
```python
adapter_output: (3, 384, 188, 188)
# 每个位置的384维特征被线性变换
```

---

## Layer 6: Sparse Tensor Reconstruction

### 作用
把dense BEV转回sparse format（因为dense_head需要SparseConvTensor）

### 输入
```python
adapter_output: (3, 384, 188, 188)  # Dense BEV

sparse_tensor_template的indices: (3503, 4)
# 这些是原始backbone输出的非零体素位置
```

### 重建过程
```python
# 从dense BEV中提取原始sparse位置的特征
for each sparse location (batch, z, y, x):
    feature = adapter_output[batch, :, y, x]  # (384,)

输出: fused_sparse_tensor
    features: (3503, 384)
    indices: (3503, 4)  # [batch, z, y, x]
    spatial_shape: [5, 188, 188]
```

### 注意
这一步只是**格式转换**，特征值不变。

### 输出
```python
SparseConvTensor(
    features: (3503, 384),
    indices: (3503, 4),
    spatial_shape: [某个shape]
)
```

---

## Layer 7: Dense Head - 多个分支

Dense head有**多个预测分支**，每个分支预测不同的属性。

### 7.1 HM分支（Heatmap）

**结构**：
```python
SubMConv2d(384, 384, kernel=3×3) + BN + ReLU  # 中间层
  → features: (3503, 384)

SubMConv2d(384, 1, kernel=1×1)  # 输出层
  → features: (3503, 1)
```

**中间层输出**：
```python
# 3×3卷积聚合周围信息
intermediate: (3503, 384)

含义：融合了每个体素与周围8个邻居的信息
```

**输出层输出**：
```python
hm_logits: (3503, 1)  → squeeze → (3503,)

数值示例（诊断数据）：
[-137.77, -125.3, -98.4, ..., -12.5, -8.2, -6.23]
           ↑ 全是负值！

每个值的含义：
hm_logits[i] = "第i个体素位置有物体中心的logit（未归一化概率）"

理想情况：
- 有物体的位置：hm_logits > 0 (sigmoid后 > 0.5)
- 背景：hm_logits < -2 (sigmoid后 < 0.12)
```

### 7.2 其他分支

**loc_x, loc_y, loc_z分支**（位置偏移 + 关键点）：
```python
SubMConv2d(384, 384, 3×3) + BN + ReLU
SubMConv2d(384, 15, 1×1)  # 1个中心点 + 14个关键点

输出: (3503, 15)
含义：
- loc_x[:, 0]: 物体中心相对于体素中心的x偏移（米）
- loc_x[:, 1:15]: 14个关键点的x坐标（相对于中心）
```

**dim分支**（尺寸）：
```python
输出: (3503, 3)
含义: [长, 宽, 高]（对数空间，需要exp还原）
```

**rot分支**（旋转）：
```python
输出: (3503, 2)
含义: [cos(θ), sin(θ)]  # 物体的朝向角度
```

**kp_vis分支**（关键点可见性）：
```python
输出: (3503, 14)
含义: 每个关键点的可见性概率（经过sigmoid）
```

### 所有分支的输出汇总
```python
pred_dict = {
    'hm': (3503,),        # Heatmap logits
    'loc_x': (3503, 15),  # X方向位置
    'loc_y': (3503, 15),  # Y方向位置
    'loc_z': (3503, 15),  # Z方向位置
    'dim': (3503, 3),     # 尺寸
    'rot': (3503, 2),     # 旋转
    'kp_vis': (3503, 14), # 关键点可见性
}
```

---

## Layer 8: Sigmoid & TopK选择

### HM Sigmoid
```python
hm_probs = sigmoid(hm_logits)

诊断数据：
input: [-137.77, ..., -6.23]
output: [0.0000, ..., 0.0020]
          ↑ 全部 < 0.3阈值
```

### TopK选择
```python
# 选择概率最高的K个位置作为候选
K = 500  # MAX_OBJ_PER_SAMPLE

# 即使所有值都<0.3，也会选top 500
scores, inds = torch.topk(hm_probs, K)

scores示例（当前情况）：
[0.0020, 0.0019, 0.0015, ..., 0.0001, 0.0000]
```

### 阈值过滤
```python
SCORE_THRESH = 0.3

mask = scores > 0.3
# 当前情况：mask全是False → 0个检测

final_boxes = boxes[mask]  # Empty!
```

---

## Layer 9: 解码成3D框 + 关键点

### 如果有通过阈值的候选

```python
for each selected voxel index i:
    # 体素中心坐标
    voxel_center = (indices[i] * voxel_size + offset)
    # 例如：voxel[750, 800] → world[30.0m, 32.0m]

    # 物体中心 = 体素中心 + 偏移
    center_x = voxel_center_x + loc_x[i, 0] * stride * voxel_size
    center_y = voxel_center_y + loc_y[i, 0] * stride * voxel_size
    center_z = loc_z[i, 0]

    # 尺寸
    l, w, h = exp(dim[i])  # 还原对数

    # 旋转角度
    angle = atan2(rot[i, 1], rot[i, 0])

    # 3D框
    box_3d = [center_x, center_y, center_z, l, w, h, angle]

    # 关键点
    for j in range(14):
        kp_x = center_x + loc_x[i, j+1] * stride * voxel_size
        kp_y = center_y + loc_y[i, j+1] * stride * voxel_size
        kp_z = loc_z[i, j+1]
        kp_visibility = kp_vis[i, j]

        keypoints[j] = [kp_x, kp_y, kp_z, kp_visibility]
```

### 最终输出格式
```python
final_pred_dict = {
    'pred_boxes': (N_detected, 7),     # [x, y, z, l, w, h, θ]
    'pred_kps': (N_detected, 14, 3),   # 14个关键点的3D坐标
    'pred_kps_vis': (N_detected, 14),  # 可见性
    'pred_scores': (N_detected,),      # 置信度
    'pred_labels': (N_detected,),      # 类别（这里都是"人"）
}

当前情况：
N_detected = 0 （因为全部 < 0.3阈值）
```

---

## 📊 完整流程总结表

| Layer | 输入 Shape | 输出 Shape | 输出含义 |
|-------|-----------|-----------|---------|
| 0. 点云 | `(N_points, 5)` | - | 原始激光点 |
| 1. VFE | `(N_points, 5)` | `(N_voxels, 5)` | 体素特征 |
| 2. Backbone3D | `(N_voxels, 5)` | `(~1200, 128)` | 高级3D特征 |
| 3. BEV | `(B, 128, 5, 188, 188)` | `(B, 384, 188, 188)` | 鸟瞰图特征 |
| 4. Transformer | `(B, 3, 384, 188, 188)` | `(B, 384, 188, 188)` | 时序融合特征 |
| 5. Adapter | `(B, 384, 188, 188)` | `(B, 384, 188, 188)` | 适配后特征 |
| 6. Sparse重建 | Dense BEV | `(3503, 384)` sparse | Sparse格式 |
| 7. Dense Head | `(3503, 384)` | `pred_dict` | 各分支预测 |
| 8. Sigmoid & TopK | `hm: (3503,)` | `scores: (500,)` | 候选位置 |
| 9. 解码 | `scores, loc, dim, rot, kp` | `(N, 7), (N, 14, 3)` | 3D框+关键点 |

---

## 🔑 关键发现（基于诊断数据）

### 1. Transformer的归一化效应
```
输入BEV特征（理论）：mean ≈ -15, std ≈ 20
Transformer输出：    mean ≈ -0.005, std ≈ 0.986

→ Transformer内部的LayerNorm强制归一化了特征！
```

### 2. Adapter的错误偏移
```
Transformer输出：mean ≈ -0.005, std ≈ 0.986
Adapter输出：    mean ≈ -1.287, std ≈ 1.415

→ Adapter引入了-1.3的偏移（应该是恒等变换）
→ 变换公式：output ≈ 1.435 × input - 1.28
```

### 3. Dense Head的分布不匹配
```
Dense Head期望输入：mean ≈ -15, std ≈ 20（预训练时）
实际输入：          mean ≈ -1.09, std ≈ 1.57

→ 分布完全不同！
→ HM卷积权重把(-1.09)放大到(-49)，全是负值
→ sigmoid(-49) ≈ 0 << 0.3阈值
```

### 4. 检测失败的原因链
```
1. Transformer归一化
   ↓ mean: -15 → -0.005

2. Adapter错误偏移
   ↓ mean: -0.005 → -1.287

3. Dense Head权重不匹配
   ↓ mean: -1.287 → -49 (HM logits)

4. Sigmoid激活
   ↓ sigmoid(-49) ≈ 0

5. 阈值过滤
   ↓ 0 < 0.3 → 被过滤

结果: 0个检测
```

---

## 💡 解决方向

### 方向1：修复Adapter初始化
确保Adapter是恒等变换（已在代码中修复）

### 方向2：残差连接
保留原始BEV特征，只添加Transformer的增强信息

### 方向3：微调Dense Head
让HM卷积权重适应新的特征分布（但有风险）

---

## 参考文件

- 完整代码：`pcdet/models/detectors/voxelnext_kp.py`
- Dense Head：`pcdet/models/dense_heads/voxelnext_head_kp_merge.py`
- 诊断工具：`pcdet/models/model_utils/feature_diagnostics.py`
- 配置文件：`tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml`
