import pickle
import re
from pathlib import Path
from collections import defaultdict

def build_sequences_with_gap_detection(all_frame_infos, sequence_length=5):
    sequences = []
    infos_by_video = defaultdict(list)
    pattern = re.compile(r'(.*) \(Frame (\d+)\)')

    for info in all_frame_infos:
        match = pattern.match(info['frame_id'])
        if match:
            video_id, frame_num_str = match.groups()
            info['frame_num'] = int(frame_num_str)
            infos_by_video[video_id].append(info)

    frame_gap_threshold = 100

    for video_id, frame_infos in infos_by_video.items():
        frame_infos.sort(key=lambda e: e['frame_num'])
        sub_segments = []
        current_segment = [frame_infos[0]]

        for i in range(1, len(frame_infos)):
            frame_gap = frame_infos[i]['frame_num'] - frame_infos[i-1]['frame_num']
            if frame_gap > frame_gap_threshold:
                sub_segments.append(current_segment)
                current_segment = [frame_infos[i]]
                print(f'检测到片段边界: {video_id}')
                print(f'  帧 {frame_infos[i-1]["frame_num"]} -> {frame_infos[i]["frame_num"]} (间隔={frame_gap})')
            else:
                current_segment.append(frame_infos[i])
        sub_segments.append(current_segment)

        print(f'\n{video_id} 被分割成 {len(sub_segments)} 个子片段:')
        for idx, seg in enumerate(sub_segments):
            seg_frame_nums = [s['frame_num'] for s in seg]
            print(f'  子片段 {idx+1}: {len(seg)} 帧, 帧号范围 {seg_frame_nums[0]} ~ {seg_frame_nums[-1]}')

        for segment in sub_segments:
            for i in range(len(segment) - sequence_length + 1):
                sequences.append(segment[i: i + sequence_length])

    return sequences


# 测试训练集
print("="*60)
print("测试训练集")
print("="*60)

info_path = Path('data/custom/custom_infos_train.pkl')
with open(info_path, 'rb') as f:
    train_infos = pickle.load(f)

print(f"加载 {len(train_infos)} 帧\n")

sequences = build_sequences_with_gap_detection(train_infos, sequence_length=5)

print(f"\n生成 {len(sequences)} 个序列")

# 验证
cross_boundary_count = 0
for i, seq in enumerate(sequences):
    frame_nums = [info['frame_num'] for info in seq]
    max_gap = max([frame_nums[j+1] - frame_nums[j] for j in range(len(frame_nums)-1)])
    if max_gap > 100:
        cross_boundary_count += 1

if cross_boundary_count == 0:
    print("✅ 训练集所有序列都没有跨越片段边界！")
else:
    print(f"❌ 训练集发现 {cross_boundary_count} 个跨边界序列！")
