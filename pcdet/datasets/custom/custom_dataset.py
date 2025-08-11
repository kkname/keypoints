# pcdet/datasets/custom/custom_dataset.py (Final Cleaned Version)

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from ...ops.iou3d_nms import iou3d_nms_utils
from ..dataset import DatasetTemplate
import torch

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_path = Path(self.root_path)

        # 直接加载infos.pkl文件
        self.infos = []
        self.include_custom_data(self.mode)

        # 从配置中读取关键点数量
        self.num_keypoints = self.dataset_cfg.get('NUM_KEYPOINTS', 14)

        if self.logger is not None:
            self.logger.info(f'Total samples for CUSTOM dataset ({self.split}): {len(self.infos)}')

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        重写此函数，以从零开始构建包含关键点信息的预测字典。
        """
        annos = []
        for index, pred_dict in enumerate(pred_dicts):
            # 获取当前样本的 frame_id
            frame_id = batch_dict['frame_id'][index]

            # 从预测结果中提取所有需要的信息
            pred_scores = pred_dict['pred_scores'].cpu().numpy()
            pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
            pred_labels = pred_dict['pred_labels'].cpu().numpy()
            pred_kps = pred_dict.get('pred_kps').cpu().numpy() if 'pred_kps' in pred_dict else None
            pred_kps_vis = pred_dict.get('pred_kps_vis').cpu().numpy() if 'pred_kps_vis' in pred_dict else None

            # 为当前样本中的每一个预测框创建标注字典
            for i in range(len(pred_boxes)):
                anno = {
                    'frame_id': frame_id,
                    'name': class_names[pred_labels[i] - 1],
                    'score': pred_scores[i],
                    'location': pred_boxes[i, 0:3],
                    'dimensions': pred_boxes[i, 3:6],
                    'rotation_y': pred_boxes[i, 6],
                    'alpha': -np.arctan2(-pred_boxes[i, 1], pred_boxes[i, 0]) + pred_boxes[i, 6],
                    'bbox': np.array([0, 0, 50, 50]),  # bbox 2d, can be dummy
                    'truncated': 0.0,
                    'occluded': 0,
                }

                # 将关键点和可见性信息添加进去
                if pred_kps is not None:
                    anno['keypoints'] = pred_kps[i]

                if pred_kps_vis is not None:
                    anno['keypoints_visible'] = pred_kps_vis[i].flatten()

                annos.append(anno)

        # 按 frame_id 对所有预测结果进行分组
        final_annos = {}
        for anno in annos:
            frame_id = anno['frame_id']
            if frame_id not in final_annos:
                final_annos[frame_id] = []
            final_annos[frame_id].append(anno)

        # 返回评估函数所期望的格式：一个列表，每个元素是对应样本的所有预测框列表
        return [final_annos.get(frame_id, []) for frame_id in batch_dict['frame_id']]

    def include_custom_data(self, mode):
        # 加载由我们脚本生成的pkl文件
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

    def get_lidar(self, lidar_path):
        # 假设点云文件是 .bin 格式，每点4个特征 (x, y, z, intensity)
        return np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = self.infos[index].copy()

        points_path = self.root_path / info['point_cloud']['lidar_path']
        points = self.get_lidar(points_path)

        input_dict = {'points': points, 'frame_id': info['frame_id']}

        if 'annos' in info:
            annos = info['annos']

            if self.dataset_cfg.get('LABEL_MAPPING', None):
                for k, v in self.dataset_cfg.LABEL_MAPPING.items():
                    annos['name'][annos['name'] == k] = v

            gt_mask = np.array([n in self.class_names for n in annos['name']], dtype=np.bool_)

            if self.dataset_cfg.get('REMOVE_BOXES_WITHOUT_KEYPOINTS', False) and 'keypoints' in annos:
                keypoints_vis_sum = annos['keypoints_visible'].sum(axis=1)
                keypoints_mask = keypoints_vis_sum > 0
                gt_mask = np.logical_and(gt_mask, keypoints_mask)

            annos['name'] = annos['name'][gt_mask]
            loc = annos['location'][gt_mask]
            dims = annos['dimensions'][gt_mask]
            rots = annos['rotation_y'][gt_mask]

            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({'gt_names': annos['name'], 'gt_boxes': gt_boxes_lidar})

            if 'keypoints' in annos:
                # 我们从annos中同时获取 keypoints 和 visibility 数据
                keypoints = annos['keypoints'][gt_mask].astype(np.float32).reshape(-1, self.num_keypoints, 3)
                keypoints_visible = annos['keypoints_visible'][gt_mask].astype(np.int32).reshape(-1, self.num_keypoints)

                input_dict.update({
                    'keypoint_location': keypoints,
                    'keypoint_visibility': keypoints_visible,
                    # --- 关键修正：新增 keypoint_mask 字段 ---
                    # 数据增强器需要这个字段。在这里，我们可以直接复用visibility作为mask。
                    'keypoint_mask': keypoints_visible
                })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    # def evaluation(self, det_annos, class_names, **kwargs):
    #     if 'custom_kp' not in self.dataset_cfg.EVAL_METRIC:
    #         # 如果评估指标不是custom_kp，则调用父类的标准评估方法
    #         return super().evaluation(det_annos, class_names, **kwargs)
    #
    #     logger = kwargs.get('logger', None)
    #     # 直接从 self.infos 中构建真值列表
    #     gt_annos = [info['annos'] for info in self.infos]
    #
    #     if logger:
    #         logger.info('**********************Start Custom Keypoint (MPJPE) Evaluation**********************')
    #
    #     # 确保预测和真值的样本数量一致
    #     if len(det_annos) != len(gt_annos):
    #         if logger:
    #             logger.warning(
    #                 f"Prediction and ground truth have different number of samples. Pred: {len(det_annos)}, GT: {len(gt_annos)}")
    #         # Handle mismatch, maybe return empty or log error
    #         return {}, 'Annotation count mismatch'
    #
    #     mpjpe_list = []
    #
    #     # 遍历每个样本
    #     for i in range(len(det_annos)):
    #         pred_annos_per_sample = det_annos[i]
    #         gt_anno_per_sample = gt_annos[i]
    #
    #         # 如果当前样本没有预测结果或没有真值，则跳过
    #         if not pred_annos_per_sample or 'keypoints' not in gt_anno_per_sample or len(
    #                 gt_anno_per_sample['keypoints']) == 0:
    #             continue
    #
    #         # 注意：这里的逻辑被简化了，它假设每个样本只评估第一个预测物体和第一个真值物体的匹配。
    #         # 一个完整的评估系统需要基于IoU进行复杂的匹配。
    #         try:
    #             pred_kps = np.array(pred_annos_per_sample[0]['keypoints'])
    #
    #             # 【核心修正】直接从真值字典中获取keypoints，并取第一个物体的关键点
    #             gt_kps = np.array(gt_anno_per_sample['keypoints'][0])
    #             gt_kps_vis = np.array(gt_anno_per_sample['keypoints_visible'][0])
    #
    #             # 计算误差
    #             error = np.linalg.norm(pred_kps - gt_kps, axis=-1)
    #
    #             # 只在可见的关键点上计算误差
    #             vis_mask = (gt_kps_vis == 1)
    #             if np.any(vis_mask):
    #                 vis_error = error[vis_mask]
    #                 mpjpe_list.append(np.mean(vis_error))
    #
    #         except (KeyError, IndexError) as e:
    #             if logger:
    #                 logger.warning(f"Skipping a sample due to data format error: {e}")
    #             continue
    #
    #     # 计算最终的平均误差，并转换为厘米
    #     total_mpjpe = np.mean(mpjpe_list) * 100 if len(mpjpe_list) > 0 else 0.0
    #
    #     result_str = f'\nMPJPE (cm): {total_mpjpe:.2f}\n'
    #     result_dict = {'MPJPE': total_mpjpe}
    #
    #     if logger is not None:
    #         logger.info(result_str)
    #         logger.info('**********************End Custom Keypoint Evaluation**********************')
    #
    #     return result_str, result_dict
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'custom_kp' not in self.dataset_cfg.EVAL_METRIC:
            return super().evaluation(det_annos, class_names, **kwargs)

        logger = kwargs.get('logger', None)

        # 1. 加载所有真值
        gt_annos = [info['annos'] for info in self.infos]

        if logger:
            logger.info('**********************Start Custom Evaluation**********************')

        # 2. 准备用于mAP计算的容器
        ap_dict = {}
        for cur_class in class_names:
            ap_dict[cur_class] = {
                'tp': [],
                'fp': [],
                'gt': 0
            }

        # 3. 准备用于MPJPE计算的容器
        mpjpe_list = []

        # 4. 核心匹配循环：遍历每一个样本
        for i in range(len(det_annos)):
            pred_annos_per_sample = det_annos[i]
            gt_annos_per_sample = gt_annos[i]

            # 统计当前样本的真值数量
            for gt_box in gt_annos_per_sample.get('name', []):
                ap_dict[gt_box]['gt'] += 1

            # 按类别分开处理
            for cur_class in class_names:
                # 筛选当前类别的预测和真值
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

                # 使用GPU进行高效的3D IoU计算
                pred_boxes_tensor = torch.from_numpy(pred_boxes_cls).cuda()
                gt_boxes_tensor = torch.from_numpy(gt_boxes_cls).cuda()
                iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes_tensor, gt_boxes_tensor).cpu().numpy()

                # 初始化匹配状态
                num_preds = pred_boxes_cls.shape[0]
                num_gts = gt_boxes_cls.shape[0]
                gt_matched = np.zeros(num_gts)

                # 贪心匹配：按置信度从高到低遍历每个预测
                sorted_indices = np.argsort(-pred_scores_cls)
                for pred_idx in sorted_indices:
                    # 找到与当前预测有最大IoU的真值
                    max_iou = -1
                    matched_gt_idx = -1
                    if iou_matrix.shape[0] > pred_idx:
                        gt_indices = np.where(iou_matrix[pred_idx, :] > 0.5)[0]  # 假设IoU阈值为0.5
                        if gt_indices.size > 0:
                            max_iou_gt_idx = np.argmax(iou_matrix[pred_idx, gt_indices])
                            max_iou = iou_matrix[pred_idx, gt_indices[max_iou_gt_idx]]
                            matched_gt_idx = gt_indices[max_iou_gt_idx]

                    # 判断是否匹配成功
                    if max_iou > 0.5 and gt_matched[matched_gt_idx] == 0:
                        ap_dict[cur_class]['tp'].append(1)
                        ap_dict[cur_class]['fp'].append(0)
                        gt_matched[matched_gt_idx] = 1

                        print("DEBUGl:A TURE POSITIVE MATCH WAS FOUND!")

                        # 对于这个成功匹配的TP，计算MPJPE
                        pred_kps = pred_kps_cls[pred_idx]
                        gt_kps = gt_kps_cls[matched_gt_idx]
                        gt_kps_vis = gt_kps_vis_cls[matched_gt_idx]

                        error = np.linalg.norm(pred_kps - gt_kps, axis=-1)
                        vis_mask = (gt_kps_vis == 1)

                        # 只打印前5个匹配对的信息，避免刷屏
                        if len(mpjpe_list) < 5:
                            mean_error_for_this_person = np.mean(error[vis_mask]) * 100 if np.any(vis_mask) else 0.0
                            print('\\n' + '=' * 50)
                            print(
                                f"DEBUG: Matched pair in frame {i} (Prediction idx {pred_idx}, GT idx {matched_gt_idx})")
                            print(f"  - Predicted Box Center: {pred_boxes_cls[pred_idx][:3]}")
                            print(f"  - Ground Truth Box Center: {gt_boxes_cls[matched_gt_idx][:3]}")
                            print(f"  - Calculated MPJPE for this person (cm): {mean_error_for_this_person:.2f}")
                            # 打印一个关节点对比，例如鼻子 (索引0)
                            print(f"  - Nose Pred: {pred_kps[0]}")
                            print(f"  - Nose GT:   {gt_kps[0]}")
                            print('=' * 50 + '\\n')

                        if np.any(vis_mask):
                            mpjpe_list.append(np.mean(error[vis_mask]))
                    else:
                        ap_dict[cur_class]['tp'].append(0)
                        ap_dict[cur_class]['fp'].append(1)

        # 5. 计算最终评估结果
        total_gt = sum([val['gt'] for val in ap_dict.values()])
        total_tp = sum([sum(val['tp']) for val in ap_dict.values()])

        # 计算mAP (此处简化为计算总的Precision和Recall)
        precision = total_tp / (total_tp + sum([sum(val['fp']) for val in ap_dict.values()])) if (total_tp + sum(
            [sum(val['fp']) for val in ap_dict.values()])) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        total_mpjpe = np.mean(mpjpe_list) * 100 if len(mpjpe_list) > 0 else 0.0

        result_str = f'\nRecall: {recall:.4f}, Precision: {precision:.4f}, MPJPE (cm): {total_mpjpe:.2f}\n'
        result_dict = {
            'Recall': recall,
            'Precision': precision,
            'MPJPE': total_mpjpe
        }

        if logger is not None:
            logger.info(result_str)
            logger.info('**********************End Custom Evaluation**********************')

        return result_str, result_dict


# -------------------- .pkl文件生成脚本部分 --------------------
def process_single_scene(scene_id, points_dir, labels_dir, num_point_features):
    import json
    lidar_file = points_dir / f'{scene_id}.bin'
    label_file = labels_dir / f'{scene_id}.json'

    with open(label_file, 'r') as f:
        annos_json = json.load(f)

    gt_names, gt_boxes_lidar, keypoints_list, keypoints_visible_list = [], [], [], []

    for obj in annos_json:
        gt_names.append(obj['class'])
        box = obj['box_3d']
        gt_boxes_lidar.append([*box['center'], *box['size'], box['heading']])
        keypoints_list.append(obj['keypoints'])
        keypoints_visible_list.append(obj['keypoints_visible'])

    annos = {
        'name': np.array(gt_names),
        'location': np.array(gt_boxes_lidar, dtype=np.float32)[:, 0:3],
        'dimensions': np.array(gt_boxes_lidar, dtype=np.float32)[:, 3:6],
        'rotation_y': np.array(gt_boxes_lidar, dtype=np.float32)[:, 6],
        'gt_boxes_lidar': np.array(gt_boxes_lidar, dtype=np.float32),
        'keypoints': np.array(keypoints_list, dtype=np.float32),
        'keypoints_visible': np.array(keypoints_visible_list, dtype=np.int32),
    }

    info = {
        'point_cloud': {
            'num_features': num_point_features,
            'lidar_path': str(Path('points') / f'{scene_id}.bin')
        },
        'frame_id': scene_id,
        'annos': annos,
    }
    return info


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val']
    num_point_features = dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list.__len__()

    for split in splits:
        split_file = data_path / 'ImageSets' / f'{split}.txt'
        if not split_file.exists():
            print(f'Split file not found, skipping: {split_file}')
            continue

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f.readlines()]

        all_infos = []
        for scene_id in tqdm(scene_ids, desc=f'Processing {split}'):
            info = process_single_scene(scene_id, data_path / 'points', data_path / 'labels', num_point_features)
            all_infos.append(info)

        output_file = save_path / f'custom_infos_{split}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(all_infos, f)
        print(f'Custom info {split} file is saved to {output_file}')


if __name__ == '__main__':
    import sys
    import yaml
    from easydict import EasyDict

    if len(sys.argv) > 1 and sys.argv[1] == 'create_custom_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=dataset_cfg.CLASS_NAMES,
            data_path=ROOT_DIR / dataset_cfg.DATA_PATH,
            save_path=ROOT_DIR / dataset_cfg.DATA_PATH
        )