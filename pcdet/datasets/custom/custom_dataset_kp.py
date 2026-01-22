# key-point/pcdet/datasets/custom/custom_dataset.py
# (Final Modified Version with Keypoint Support and Info Generation Script)

import numpy as np
import pickle
import torch
import copy
from pathlib import Path
from ...utils import box_utils
from ..dataset import DatasetTemplate
from ...ops.iou3d_nms import iou3d_nms_utils


CUSTOM_KEYPOINT_NAME_TO_ID = {
    'nose':0,
    'left_shoulder': 1,
    'right_shoulder': 2,
    'left_elbow': 3,
    'right_elbow': 4,
    'left_wrist': 5,
    'right_wrist': 6,
    'left_hip': 7,
    'right_hip': 8,
    'left_knee': 9,
    'right_knee': 10,
    'left_ankle': 11,
    'right_ankle': 12,
    'head': 13,
}

CUSTOM_KEYPOINT_ID_TO_NAME = {v: k for k, v in CUSTOM_KEYPOINT_NAME_TO_ID.items()}

class CustomDatasetKP(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split == 'train' else 'testing')

        # 直接加载infos.pkl文件，不再需要sample_id_list
        self.infos = []
        self.include_custom_data(self.mode)

        # -- KEYPOINT MODIFICATION --
        # 从配置中读取关键点数量
        self.num_keypoints = self.dataset_cfg.get('NUM_KEYPOINTS', 14)

    def include_custom_data(self, mode):
        info_path_list = self.dataset_cfg.INFO_PATH[mode]
        for info_path in info_path_list:
            info_path = self.root_path / info_path
            if not info_path.exists():
                if self.logger is not None:
                    self.logger.warning(f"Info file not found: {info_path}")
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                self.infos.extend(infos)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        重写此函数，以从零开始构建包含所有必要信息的、结构正确的预测字典。
        Args:
            pred_dicts: list of pred_dicts, e.g.,
                [
                    {
                        'pred_boxes': (N, 7), 'pred_scores': (N), 'pred_labels': (N),
                        'pred_kps': (N, 14, 3), 'pred_kps_vis': (N, 14, 1)
                    }, ...
                ]
        Returns:
            annos: A list of list of dicts.
                The outer list corresponds to samples in batch.
                The inner list corresponds to detected boxes in a sample.
        """
        annos = []
        for index, pred_dict in enumerate(pred_dicts):
            single_pred_list = []
            frame_id = batch_dict['frame_id'][index]

            pred_scores = pred_dict['pred_scores'].cpu().numpy()
            pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
            pred_labels = pred_dict['pred_labels'].cpu().numpy()
            pred_kps = pred_dict['pred_kps'].cpu().numpy()
            pred_kps_vis = pred_dict['pred_kps_vis'].cpu().numpy()

            for i in range(len(pred_boxes)):
                # 为当前样本的每一个预测框创建一个标准anno字典
                anno = {
                    'frame_id': frame_id,
                    'name': class_names[pred_labels[i] - 1],
                    'score': pred_scores[i],
                    'boxes_lidar': pred_boxes[i],
                    'keypoints': pred_kps[i],
                    'keypoints_visible': pred_kps_vis[i].flatten()
                }
                single_pred_list.append(anno)

            annos.append(single_pred_list)

        return annos

    def get_lidar(self, lidar_path):
        # 假设点云文件是 .bin 格式，每点4个特征 (x, y, z, intensity)
        return np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    # --- 必须实现的 __getitem__ 方法 ---
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = self.infos[index].copy()
        points_path = self.root_path / info['point_cloud']['lidar_path']
        points = self.get_lidar(points_path)
        input_dict = {'points': points, 'frame_id': info['frame_id']}

        if 'annos' in info:
            annos = info['annos']
            gt_mask = np.array([n in self.class_names for n in annos['name']], dtype=bool)

            # 过滤掉不需要的物体
            annos = {key: val[gt_mask] for key, val in annos.items() if isinstance(val, np.ndarray)}

            input_dict['gt_names'] = annos['name']
            input_dict['gt_boxes'] = annos['gt_boxes_lidar']

            if 'keypoints' in annos:
                # 修正键名以匹配collate_batch的要求
                input_dict['keypoint_location'] = annos['keypoints'].reshape(-1, self.num_keypoints, 3)
                input_dict['keypoint_visibility'] = annos['keypoints_visible'].reshape(-1, self.num_keypoints)
                # 提供data_augmentor需要的keypoint_mask
                input_dict['keypoint_mask'] = annos['keypoints_visible'].reshape(-1, self.num_keypoints)

        # 调用父类的prepare_data进行数据增强和处理
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'custom_kp' not in self.dataset_cfg.EVAL_METRIC:
            self.logger.info('Skip keypoint evaluation since the EVAL_METRIC is not "custom_kp".')
            return 'Evaluation metric %s not supported.' % self.dataset_cfg.EVAL_METRIC, {}

        logger = kwargs.get('logger', None)
        if logger:
            logger.info('**********************Start Custom Evaluation (Visible/Invisible MPJPE)**********************')

        gt_annos = [info['annos'] for info in self.infos]

        vis_error_sum = np.zeros(self.num_keypoints, dtype=np.float64)
        vis_count = np.zeros(self.num_keypoints, dtype=np.int64)
        invis_error_sum = np.zeros(self.num_keypoints, dtype=np.float64)
        invis_count = np.zeros(self.num_keypoints, dtype=np.int64)

        for frame_idx in range(len(det_annos)):
            pred_annos_per_sample = det_annos[frame_idx]
            gt_annos_per_sample = gt_annos[frame_idx]

            for cur_class in class_names:
                pred_boxes_cls = np.array(
                    [anno['boxes_lidar'] for anno in pred_annos_per_sample if anno['name'] == cur_class])
                pred_scores_cls = np.array(
                    [anno['score'] for anno in pred_annos_per_sample if anno['name'] == cur_class])
                pred_kps_cls = np.array(
                    [anno['keypoints'] for anno in pred_annos_per_sample if anno['name'] == cur_class])

                gt_boxes_cls_mask = (gt_annos_per_sample['name'] == cur_class)
                gt_boxes_cls = gt_annos_per_sample['gt_boxes_lidar'][gt_boxes_cls_mask]
                gt_kps_cls = gt_annos_per_sample['keypoints'][gt_boxes_cls_mask]
                gt_kps_vis_cls = gt_annos_per_sample['keypoints_visible'][gt_boxes_cls_mask]

                if pred_boxes_cls.shape[0] == 0 or gt_boxes_cls.shape[0] == 0:
                    continue

                pred_boxes_tensor = torch.from_numpy(pred_boxes_cls).cuda()
                gt_boxes_tensor = torch.from_numpy(gt_boxes_cls).cuda()
                iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes_tensor, gt_boxes_tensor).cpu().numpy()

                gt_matched = np.zeros(gt_boxes_cls.shape[0])
                sorted_indices = np.argsort(-pred_scores_cls)

                for pred_idx in sorted_indices:
                    max_iou = -1
                    matched_gt_idx = -1
                    if iou_matrix.shape[0] > pred_idx:
                        gt_indices = np.where(iou_matrix[pred_idx, :] > 0.5)[0]
                        if gt_indices.size > 0:
                            max_iou_gt_idx = np.argmax(iou_matrix[pred_idx, gt_indices])
                            max_iou = iou_matrix[pred_idx, gt_indices[max_iou_gt_idx]]
                            matched_gt_idx = gt_indices[max_iou_gt_idx]

                    if max_iou > 0.5 and gt_matched[matched_gt_idx] == 0:
                        gt_matched[matched_gt_idx] = 1

                        pred_kps = pred_kps_cls[pred_idx]
                        gt_kps = gt_kps_cls[matched_gt_idx]
                        gt_kps_vis = gt_kps_vis_cls[matched_gt_idx]

                        error = np.linalg.norm(pred_kps - gt_kps, axis=-1)

                        visible_mask = (gt_kps_vis == 1)
                        invisible_mask = (gt_kps_vis == 0)

                        vis_error_sum += error * visible_mask
                        vis_count += visible_mask
                        invis_error_sum += error * invisible_mask
                        invis_count += invisible_mask

        result_str = '\n'
        result_dict = {}

        def generate_report(title, error_sum, count, result_dict):
            report_str = f'--- MPJPE Evaluation ({title}) ---\n'
            per_joint_mpjpe = np.divide(error_sum, count, out=np.zeros_like(error_sum), where=count != 0)
            overall_mpjpe = np.sum(error_sum) / np.sum(count) if np.sum(count) > 0 else 0.0
            per_joint_mpjpe_cm = per_joint_mpjpe * 100
            overall_mpjpe_cm = overall_mpjpe * 100

            report_str += 'Per-Keypoint MPJPE (cm):\n'
            for i in range(self.num_keypoints):
                keypoint_name = CUSTOM_KEYPOINT_ID_TO_NAME.get(i, f'unknown_{i}')
                report_str += f'  - {keypoint_name:15s}: {per_joint_mpjpe_cm[i]:.2f}\n'
                result_dict[f'MPJPE_{title}_{keypoint_name}'] = per_joint_mpjpe_cm[i]

            report_str += '-----------------------------------\n'
            report_str += f'Overall MPJPE (cm): {overall_mpjpe_cm:.2f}\n\n'
            result_dict[f'MPJPE_Overall_{title}_cm'] = overall_mpjpe_cm
            return report_str, result_dict

        result_str_vis, result_dict = generate_report('Visible_Only', vis_error_sum, vis_count, result_dict)
        result_str += result_str_vis
        result_str_invis, result_dict = generate_report('Invisible_Only', invis_error_sum, invis_count, result_dict)
        result_str += result_str_invis

        all_error_sum = vis_error_sum + invis_error_sum
        all_count = vis_count + invis_count
        result_str_all, result_dict = generate_report('All_Points', all_error_sum, all_count, result_dict)
        result_str += result_str_all

        if logger is not None:
            logger.info(result_str)
            logger.info('**********************End Custom Evaluation**********************')

        return result_str, result_dict


# -------------------- .pkl文件生成脚本部分 --------------------
# 我们将原来的main执行块替换为我们自己的生成脚本
# 您可以根据自己的标注格式修改 process_single_scene 函数

def process_single_scene(scene_id, points_dir, labels_dir, num_point_features):
    """
    处理单帧数据并生成符合pcdet格式的字典
    """
    import json
    lidar_file = points_dir / f'{scene_id}.bin'
    label_file = labels_dir / f'{scene_id}.json'

    # 载入标注json文件 (这是你需要根据自己格式修改的核心)
    with open(label_file, 'r') as f:
        annos_json = json.load(f)

    gt_names = []
    gt_boxes_lidar = []
    keypoints_list = []
    keypoints_visible_list = []

    for obj in annos_json:
        gt_names.append(obj['class'])
        box = obj['box_3d']
        # 顺序: cx, cy, cz, dx, dy, dz, heading
        gt_boxes_lidar.append([*box['center'], *box['size'], box['heading']])
        keypoints_list.append(obj['keypoints'])
        keypoints_visible_list.append(obj['keypoints_visible'])

    gt_names_np = np.array(gt_names)
    gt_boxes_lidar_np = np.array(gt_boxes_lidar, dtype=np.float32)
    keypoints_np = np.array(keypoints_list, dtype=np.float32)
    keypoints_visible_np = np.array(keypoints_visible_list, dtype=np.int32)

    # 组装pcdet所需的 'annos' 字典
    annos = {
        'name': gt_names_np,
        'location': gt_boxes_lidar_np[:, 0:3],
        'dimensions': gt_boxes_lidar_np[:, 3:6],
        'rotation_y': gt_boxes_lidar_np[:, 6],
        'gt_boxes_lidar': gt_boxes_lidar_np,
        'keypoints': keypoints_np,
        'keypoints_visible': keypoints_visible_np,
    }

    # 组装最终的 info 字典
    info = {
        'point_cloud': {
            'num_features': num_point_features,
            'lidar_path': str(lidar_file.relative_to(points_dir.parent))  # 使用相对路径
        },
        'frame_id': scene_id,
        'annos': annos,
    }

    return info


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    from tqdm import tqdm
    dataset = CustomDatasetKP(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False
    )
    train_split, val_split = 'train', 'val'
    num_point_features = 4  # 根据你的点云修改

    train_infos = []
    val_infos = []

    # 处理训练集
    train_split_file = data_path / 'ImageSets' / (train_split + '.txt')
    with open(train_split_file, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    for scene_id in tqdm(train_ids, desc=f'Processing {train_split}'):
        info = process_single_scene(scene_id, data_path / 'points', data_path / 'labels', num_point_features)
        train_infos.append(info)

    train_filename = save_path / f'custom_infos_{train_split}.pkl'
    with open(train_filename, 'wb') as f:
        pickle.dump(train_infos, f)
    print(f'Custom info train file is saved to {train_filename}')

    # 处理验证集
    val_split_file = data_path / 'ImageSets' / (val_split + '.txt')
    with open(val_split_file, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    for scene_id in tqdm(val_ids, desc=f'Processing {val_split}'):
        info = process_single_scene(scene_id, data_path / 'points', data_path / 'labels', num_point_features)
        val_infos.append(info)

    val_filename = save_path / f'custom_infos_{val_split}.pkl'
    with open(val_filename, 'wb') as f:
        pickle.dump(val_infos, f)
    print(f'Custom info val file is saved to {val_filename}')


if __name__ == '__main__':
    import sys
    import yaml
    from easydict import EasyDict

    if len(sys.argv) > 1 and sys.argv[1] == 'create_custom_infos':
        # 使用你自己的数据集配置文件
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Human'],
            data_path=ROOT_DIR / 'data' / 'custom',  # 假设数据在 data/custom 目录下
            save_path=ROOT_DIR / 'data' / 'custom'
        )