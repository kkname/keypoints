import pickle
from pathlib import Path
import re
from collections import defaultdict

# 检查训练集和验证集
for split in ['train', 'val']:
    info_path = Path(f'data/custom/custom_infos_{split}.pkl')

    if not info_path.exists():
        print(f"{split}.pkl 不存在，跳过\n")
        continue

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    print(f"{'='*60}")
    print(f"{split.upper()} 数据集:")
    print(f"{'='*60}")
    print(f"总帧数: {len(infos)}\n")

    pattern = re.compile(r'(.*) \(Frame (\d+)\)')

    # 按片段统计
    segments = defaultdict(list)
    for info in infos:
        match = pattern.match(info['frame_id'])
        if match:
            segment_name, frame_num = match.groups()
            segments[segment_name].append(int(frame_num))

    print(f"片段数量: {len(segments)}\n")

    for segment_name, frame_nums in sorted(segments.items()):
        frame_nums.sort()
        print(f"片段: {segment_name}")
        print(f"  帧数: {len(frame_nums)}")
        print(f"  帧号范围: {frame_nums[0]} ~ {frame_nums[-1]}")
        print(f"  前5帧: {frame_nums[:5]}")
        print(f"  后5帧: {frame_nums[-5:]}")
        print()

print("\n" + "="*60)
print("你提到的3个片段:")
print("="*60)
print("1. 2025-06-12-17-09-17-RS-0-Data: 195-3310")
print("2. 2025-06-12-17-15-15-RS-0-Data: 675-910")
print("3. 2025-06-12-17-15-15-RS-0-Data: 2075-2460")
print("\n注意: 片段2和3的前缀相同，但帧号范围不同")
