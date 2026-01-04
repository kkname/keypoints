import pickle
from pathlib import Path
import re

# 检查训练集的帧号连续性
info_path = Path('data/custom/custom_infos_train.pkl')

with open(info_path, 'rb') as f:
    infos = pickle.load(f)

pattern = re.compile(r'(.*) \(Frame (\d+)\)')

# 提取所有帧号
frame_numbers = []
for info in infos:
    match = pattern.match(info['frame_id'])
    if match:
        frame_num = int(match.groups()[1])
        frame_numbers.append(frame_num)

frame_numbers.sort()

print(f"总帧数: {len(frame_numbers)}")
print(f"帧号范围: {frame_numbers[0]} ~ {frame_numbers[-1]}")
print(f"前10个帧号: {frame_numbers[:10]}")
print(f"后10个帧号: {frame_numbers[-10:]}\n")

# 检测帧号间隔（gap）
gaps = []
for i in range(len(frame_numbers) - 1):
    diff = frame_numbers[i+1] - frame_numbers[i]
    if diff > 5:  # 如果间隔大于5，认为是片段边界
        gaps.append({
            'position': i,
            'before_frame': frame_numbers[i],
            'after_frame': frame_numbers[i+1],
            'gap_size': diff
        })

if gaps:
    print(f"检测到 {len(gaps)} 个可能的片段边界:\n")
    for idx, gap in enumerate(gaps):
        print(f"边界 {idx+1}:")
        print(f"  位置: 第 {gap['position']} 帧之后")
        print(f"  帧号跳跃: {gap['before_frame']} -> {gap['after_frame']}")
        print(f"  间隔: {gap['gap_size']} 帧\n")

    print(f"基于边界推断的片段数量: {len(gaps) + 1}")

    # 统计每个片段的帧数
    segment_sizes = []
    prev_pos = 0
    for gap in gaps:
        segment_sizes.append(gap['position'] + 1 - prev_pos)
        prev_pos = gap['position'] + 1
    segment_sizes.append(len(frame_numbers) - prev_pos)

    print("\n每个片段的帧数:")
    for i, size in enumerate(segment_sizes):
        print(f"  片段 {i+1}: {size} 帧")
else:
    print("未检测到帧号间隔，所有帧号都连续（间隔<=5帧）")
