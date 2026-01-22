import pickle
import numpy as np

# --- 修改这里 ---
# 换成您正在使用的 .pkl 文件的绝对路径
INFO_FILE_PATH = 'data/custom/custom_infos_val.pkl'
# ---------------

with open(INFO_FILE_PATH, 'rb') as f:
    infos = pickle.load(f)

is_valid = True
for i, info in enumerate(infos):
    if 'annos' not in info:
        continue

    gt_boxes = info['annos'].get('gt_boxes_lidar', np.array([]))

    if gt_boxes.size == 0:
        continue

    # 检查每个box是否都有7个值
    if gt_boxes.shape[1] != 7:
        print(f"!!! Error Found in Frame {i} !!!")
        print(f"  - Frame ID (from pkl): {info.get('frame_id', 'N/A')}")
        print(f"  - Ground Truth Box Shape: {gt_boxes.shape}")
        print(f"  - Corrupted Data: \n{gt_boxes}")
        print("-" * 30)
        is_valid = False

if is_valid:
    print("✅ Congratulations! Your .pkl file seems to be correctly formatted.")
else:
    print("\n❌ Problem Found. Please check the frames listed above in your dataset and data generation script.")