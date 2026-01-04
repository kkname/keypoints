import pickle
from pathlib import Path
import re

# 检查验证集的间隔
info_path = Path('data/custom/custom_infos_val.pkl')

with open(info_path, 'rb') as f:
    infos = pickle.load(f)

pattern = re.compile(r'(.*) \(Frame (\d+)\)')

# 提取所有帧号
frame_nums = []
for info in infos:
    match = pattern.match(info['frame_id'])
    if match:
        frame_nums.append(int(match.groups()[1]))

frame_nums.sort()

print(f"验证集总帧数: {len(frame_nums)}")
print(f"帧号范围: {frame_nums[0]} ~ {frame_nums[-1]}\n")

# 检查所有间隔
large_gaps = []
for i in range(len(frame_nums) - 1):
    diff = frame_nums[i+1] - frame_nums[i]
    if diff > 100:  # 间隔大于100，肯定是片段边界
        large_gaps.append({
            'index': i,
            'before': frame_nums[i],
            'after': frame_nums[i+1],
            'gap': diff
        })

if large_gaps:
    print(f"❌ 发现 {len(large_gaps)} 个巨大间隔（片段边界）:\n")
    for idx, gap in enumerate(large_gaps):
        print(f"边界 {idx+1}:")
        print(f"  位置: 第 {gap['index']} 帧和第 {gap['index']+1} 帧之间")
        print(f"  帧号跳跃: {gap['before']} -> {gap['after']}")
        print(f"  间隔: {gap['gap']} 帧")
        print()

    # 划分子片段
    segments = []
    start_idx = 0
    for gap in large_gaps:
        end_idx = gap['index'] + 1
        segments.append(frame_nums[start_idx:end_idx])
        start_idx = end_idx
    segments.append(frame_nums[start_idx:])

    print(f"基于间隔划分的子片段数量: {len(segments)}\n")
    for i, seg in enumerate(segments):
        print(f"子片段 {i+1}:")
        print(f"  帧数: {len(seg)}")
        print(f"  帧号范围: {seg[0]} ~ {seg[-1]}")
        print()

    print("⚠️ 警告: 在序列采样时会跨越这些间隔边界！")
    print("   这会导致不连续的帧被采样到同一个序列中！")
else:
    print("✅ 验证集中没有大间隔，帧号连续")
