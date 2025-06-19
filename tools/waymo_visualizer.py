import argparse
import time
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from visual_utils import open3d_vis_utils as V

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import WaymoDatasetKP
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# python waymo_visualizer.py --ckpt CHECKPOINT
#python waymo_visualizer.py --ckpt /home/yizhi/model/VoxelKP/voxelkp_checkpoint.pth --output_format json

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml', help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='/home/yizhi/model/VoxelKP/voxelkp_checkpoint.pth')
    parser.add_argument('--data_path', type=str, default='/home/yizhi/model/VoxelKP/data/custom/', help='specify the point cloud data file or directory')
    parser.add_argument('--gt_only', action="store_true", help="show results with ground truth annotation.")
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='/home/yizhi/model/VoxelKP/data/custom/output', help='directory to save prediction results')
    parser.add_argument('--output_format', type=str, default='json', choices=['json', 'txt'], help='output file format')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def save_predictions(pred_dict, frame_idx, output_dir, format='json'):
    """
    保存预测结果到文件

    Args:
        pred_dict: 预测结果字典
        frame_idx: 帧索引
        output_dir: 输出目录
        format: 输出格式，'json'或'txt'
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取预测数据
    boxes = pred_dict['pred_boxes'].cpu().numpy() if isinstance(pred_dict['pred_boxes'], torch.Tensor) else pred_dict[
        'pred_boxes']
    keypoints = pred_dict['pred_kps'].cpu().numpy() if isinstance(pred_dict['pred_kps'], torch.Tensor) else pred_dict[
        'pred_kps']
    scores = pred_dict['pred_scores'].cpu().numpy() if isinstance(pred_dict['pred_scores'], torch.Tensor) else \
    pred_dict['pred_scores']
    labels = pred_dict['pred_labels'].cpu().numpy() if isinstance(pred_dict['pred_labels'], torch.Tensor) else \
    pred_dict['pred_labels']

    output_data = []

    # 对每个预测框和关键点进行处理
    for i in range(len(boxes)):
        box_data = {
            'box': boxes[i].tolist() if isinstance(boxes[i], np.ndarray) else boxes[i],
            'keypoints': keypoints[i].tolist() if isinstance(keypoints[i], np.ndarray) else keypoints[i],
            'score': float(scores[i]),
            'label': int(labels[i])
        }
        output_data.append(box_data)

    # 保存结果
    if format.lower() == 'json':
        output_file = os.path.join(output_dir, f'frame_{frame_idx:06d}.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    else:  # txt格式
        output_file = os.path.join(output_dir, f'frame_{frame_idx:06d}.txt')
        with open(output_file, 'w') as f:
            for obj in output_data:
                box_str = ' '.join([str(v) for v in obj['box']])
                kps_str = ' '.join([str(v) for v in np.array(obj['keypoints']).flatten()])
                f.write(f"label:{obj['label']} score:{obj['score']:.4f} box:{box_str} keypoints:{kps_str}\n")

    return output_file

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    cfg.DATA_CONFIG.REMOVE_BOXES_WITHOUT_KEYPOINTS = False
    # cfg.MODEL.DENSE_HEAD.IOU_BRANCH = False
    cfg.DATA_CONFIG.LABEL_MAPPING = None

    demo_dataset = WaymoDatasetKP(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,  # For visualizing augmentations
        root_path=Path(args.data_path), logger=logger, inference_mode=True
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    time_used = []
    count = 0

    blocking = True
    with torch.inference_mode():
        for idx, data_dict in enumerate(demo_dataset):

            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])

            load_data_to_gpu(data_dict)

            start = datetime.now()
            pred_dicts, _ = model.forward(data_dict)
            time_used.append((datetime.now() - start).total_seconds())

            count+= 1
            print(f"Time used for frame {count}:", time_used[-1], "Running Average (frames/s):", 1 / (sum(time_used) / count))

            # 保存预测结果到文件
            output_file = save_predictions(pred_dicts[0], idx, args.output_dir, args.output_format)
            logger.info(f'Saved predictions to {output_file}')

            if count == 1 or blocking:
                vis = V.draw_scenes(
                    # points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    # ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    points=data_dict['points'][:, 1:],
                    # gt_boxes=data_dict['gt_boxes'][0],
                    # gt_keypoints=data_dict['keypoint_location'][0],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_keypoints=pred_dicts[0]['pred_kps'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    points_to_keep_boxes=pred_dicts[0]['pred_boxes'] if args.gt_only else None,
                    blocking=blocking
                )

                vis.run()

                vis.poll_events()
                vis.update_renderer()

                time.sleep(1)

                vis.destroy_window()
                del vis
            else:
                # Buggy
                vis = V.update_scenes(
                    vis=vis,
                    points=data_dict['points'][:, 1:],
                    # gt_boxes=data_dict['gt_boxes'][0],
                    # gt_keypoints=data_dict['keypoint_location'][0],
                    # ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_keypoints=pred_dicts[0]['pred_kps'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    points_to_keep_boxes=pred_dicts[0]['pred_boxes'] if args.gt_only else None,
                )
                time.sleep(1)

        if not blocking:
            vis.destroy_window()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
