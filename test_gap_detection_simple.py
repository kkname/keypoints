import pickle
import re
from pathlib import Path
from collections import defaultdict

# 模拟 build_sequences 函数的逻辑
def build_sequences_with_gap_detection(all_frame_infos, sequence_length=5):
    """
    模拟修改后的 build_sequences 函数
    """
    sequences = []
    infos_by_video = defaultdict(list)
    pattern = re.compile(r'(.*) \(Frame (\d+)\)')

    for info in all_frame_infos:
        match = pattern.match(info['frame_id'])
        if match:
            video_id, frame_num_str = match.groups()
            info['frame_num'] = int(frame_num_str)
            infos_by_video[video_id].append(info)

    # 帧号间隔阈值
    frame_gap_threshold = 100

    for video_id, frame_infos in infos_by_video.items():
        frame_infos.sort(key=lambda e: e['frame_num'])

        # 检测帧号间隔，自动分割子片段
        sub_segments = []
        current_segment = [frame_infos[0]]

        for i in range(1, len(frame_infos)):
            frame_gap = frame_infos[i]['frame_num'] - frame_infos[i-1]['frame_num']
            if frame_gap > frame_gap_threshold:
                # 发现大间隔
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

        # 在每个子片段内部独立进行滑动窗口采样
        for segment in sub_segments:
            for i in range(len(segment) - sequence_length + 1):
                sequences.append(segment[i: i + sequence_length])

    return sequences


# 测试验证集
print("="*60)
print("测试验证集（有间隔问题的数据集）")
print("="*60)

info_path = Path('data/custom/custom_infos_val.pkl')
with open(info_path, 'rb') as f:
    val_infos = pickle.load(f)

print(f"加载 {len(val_infos)} 帧\n")

sequences = build_sequences_with_gap_detection(val_infos, sequence_length=5)

print(f"\n生成 {len(sequences)} 个序列\n")

# 验证所有序列是否跨边界
print("="*60)
print("验证：检查所有序列是否跨越边界")
print("="*60)

cross_boundary_count = 0
for i, seq in enumerate(sequences):
    frame_nums = [info['frame_num'] for info in seq]
    max_gap = max([frame_nums[j+1] - frame_nums[j] for j in range(len(frame_nums)-1)])
    if max_gap > 100:
        cross_boundary_count += 1
        print(f"❌ 序列 {i}: {frame_nums} (最大间隔={max_gap})")

if cross_boundary_count == 0:
    print("✅ 所有序列都没有跨越片段边界！")
    print("\n边界附近的序列示例:")
    print(f"  序列43: {[info['frame_num'] for info in sequences[43]]}")
    print(f"  序列44: {[info['frame_num'] for info in sequences[44]]}")
    if len(sequences) > 45:
        print(f"  序列45: {[info['frame_num'] for info in sequences[45]]}")
else:
    print(f"❌ 发现 {cross_boundary_count} 个跨边界序列！")
