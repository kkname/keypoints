import pickle
from pathlib import Path
import re

# 检查训练集的 frame_id 格式
info_path = Path('data/custom/custom_infos_train.pkl')

with open(info_path, 'rb') as f:
    infos = pickle.load(f)

print(f"总帧数: {len(infos)}\n")

# 提取所有 video_id
pattern = re.compile(r'(.*) \(Frame (\d+)\)')
video_ids = set()

print("前10帧的 frame_id:")
for i, info in enumerate(infos[:10]):
    frame_id = info['frame_id']
    print(f"  {i}: {frame_id}")

    match = pattern.match(frame_id)
    if match:
        video_id, frame_num = match.groups()
        video_ids.add(video_id)

print(f"\n检测到的片段数量: {len(video_ids)}")
print(f"片段名称: {sorted(video_ids)}")

# 统计每个片段的帧数
from collections import defaultdict
video_frame_counts = defaultdict(int)

for info in infos:
    match = pattern.match(info['frame_id'])
    if match:
        video_id = match.groups()[0]
        video_frame_counts[video_id] += 1

print("\n每个片段的帧数:")
for video_id, count in sorted(video_frame_counts.items()):
    print(f"  {video_id}: {count} 帧")

# 检查是否所有帧都匹配格式
unmatched = [info['frame_id'] for info in infos if not pattern.match(info['frame_id'])]
if unmatched:
    print(f"\n警告: 有 {len(unmatched)} 帧的 frame_id 格式不匹配!")
    print("前5个不匹配的:")
    for frame_id in unmatched[:5]:
        print(f"  {frame_id}")
else:
    print("\n✓ 所有帧的 frame_id 格式都正确")
