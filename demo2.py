import argparse
import glob
from pathlib import Path
import json
from datetime import datetime


try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from tools.visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--no_vis', action='store_true', help='disable visualization (headless)')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    output_dir = Path("prediction_results")
    output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_kps = pred_dicts[0].get('pred_kps', None)

            # 调用我们升级后的可视化函数，并传入关键点
            if not args.no_vis:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    ref_keypoints=pred_kps  # <-- 关键新增：将关键点数据传递给可视化函数
                )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

            # 修复变量赋值错误（移除多余的逗号，避免元组类型）
            ref_boxes = pred_dicts[0]['pred_boxes']  # 移除逗号
            ref_scores = pred_dicts[0]['pred_scores']  # 移除逗号
            ref_labels = pred_dicts[0]['pred_labels']  # 移除逗号

            # 转换为numpy数组处理
            ref_boxes_np = ref_boxes.cpu().numpy()
            ref_scores_np = ref_scores.cpu().numpy()
            ref_labels_np = ref_labels.cpu().numpy()

            json_data = []
            num_objects = len(ref_boxes_np)
            logger.info(f'Detected {num_objects} objects in this frame')

            # 获取当前处理的点云文件名（用于JSON命名）
            current_file = Path(demo_dataset.sample_file_list[idx]).stem

            for i in range(num_objects):
                # 只处理人体类别（假设标签1对应Human，根据实际配置修改）
                if ref_labels_np[i] != 1:
                    continue

                # 解析3D框数据
                box = ref_boxes_np[i]
                box_3d = {
                    "center": box[:3].tolist(),  # x, y, z
                    "size": box[3:6].tolist(),  # l, w, h
                    "heading": float(box[6]),  # 朝向角
                    "score": float(ref_scores_np[i])  # 新增置信度
                }

                # 解析关键点数据
                keypoints = []
                keypoints_visible = []
                if pred_kps is not None and i < len(pred_kps):
                    kps = pred_kps[i].cpu().numpy()  # [14, 3] 假设14个关键点
                    keypoints = kps.tolist()
                    keypoints_visible = [1] * len(kps)  # 若有可见性信息可替换

                # 构造单个人体的JSON条目
                obj_data = {
                    "class": "Human",
                    "object_id": f"person_{i:02d}",
                    "box_3d": box_3d,
                    "keypoints": keypoints,
                    "keypoints_visible": keypoints_visible
                }
                json_data.append(obj_data)

            # 导出JSON文件（使用原文件名+时间戳避免重复）
            if json_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"{current_file}_{timestamp}_Frame-{idx}.json"
                with open(output_file, "w") as f:
                    json.dump(json_data, f, indent=2)
                logger.info(f"预测结果已保存至: {output_file}")

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
