import pickle
from pathlib import Path
import re

# 检查训练集的帧号连续性
info_path = Path('data/custom/custom_infos_train.pkl')

with open(info_path, 'rb') as f:
    infos = pickle.load(f)

pattern = re.compile(r'(.*) \(Frame (\d+)\)')

# 提取所有帧号
frame_data = []
for info in infos:
    match = pattern.match(info['frame_id'])
    if match:
        frame_num = int(match.groups()[1])
        frame_data.append({
            'frame_id': info['frame_id'],
            'frame_num': frame_num
        })

frame_data.sort(key=lambda x: x['frame_num'])

print(f"总帧数: {len(frame_data)}")
print(f"帧号范围: {frame_data[0]['frame_num']} ~ {frame_data[-1]['frame_num']}\n")

# 检查所有间隔
intervals = []
for i in range(len(frame_data) - 1):
    diff = frame_data[i+1]['frame_num'] - frame_data[i]['frame_num']
    intervals.append(diff)

from collections import Counter
interval_counts = Counter(intervals)

print("帧号间隔统计:")
for interval, count in sorted(interval_counts.items()):
    print(f"  间隔 {interval}: {count} 次")

# 找出所有非5的间隔
abnormal_gaps = []
for i in range(len(frame_data) - 1):
    diff = frame_data[i+1]['frame_num'] - frame_data[i]['frame_num']
    if diff != 5:
        abnormal_gaps.append({
            'index': i,
            'before': frame_data[i]['frame_num'],
            'after': frame_data[i+1]['frame_num'],
            'gap': diff
        })

if abnormal_gaps:
    print(f"\n发现 {len(abnormal_gaps)} 个异常间隔（不是5）:")
    for gap in abnormal_gaps:
        print(f"  索引 {gap['index']}: 帧 {gap['before']} -> {gap['after']}, 间隔={gap['gap']}")
else:
    print("\n所有帧号间隔都是5，完全连续")

# 计算理论帧数
expected_frames = (frame_data[-1]['frame_num'] - frame_data[0]['frame_num']) // 5 + 1
print(f"\n理论帧数（如果完全连续）: {expected_frames}")
print(f"实际帧数: {len(frame_data)}")
print(f"缺失帧数: {expected_frames - len(frame_data)}")
