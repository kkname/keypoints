# key-point/pcdet/datasets/custom/custom_dataset.py
# (Final Modified Version with Keypoint Support and Info Generation Script)

import numpy as np
import pickle
from pathlib import Path
from ...utils import box_utils
from ..dataset import DatasetTemplate


class CustomDataset(DatasetTemplate):
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
        if self.logger is not None:
            self.logger.info('Loading Custom dataset')

        # 直接加载由我们脚本生成的pkl文件
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                self.logger.warning(f"Info file not found: {info_path}")
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                self.infos.extend(infos)

        if self.logger is not None:
            self.logger.info('Total samples for CUSTOM dataset: %d' % (len(self.infos)))

    def get_lidar(self, lidar_path):
        # 假设点云文件是 .bin 格式，每点4个特征 (x, y, z, intensity)
        # 您可以根据自己的数据格式修改这里的维度
        return np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = self.infos[index].copy()

        # 从相对路径构建绝对路径
        points_path = self.root_path / info['point_cloud']['lidar_path']
        points = self.get_lidar(points_path)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            # -- KEYPOINT MODIFICATION --
            # 移除 'DontCare' 类别 (如果您的标注中有的话)
            # annos = self.remove_dont_care(annos)

            loc = annos['location']
            dims = annos['dimensions']
            rots = annos['rotation_y']
            names = annos['name']

            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': names,
                'gt_boxes': gt_boxes_lidar,
            })

            if 'keypoints' in annos:
                keypoints = annos['keypoints']
                keypoints_visible = annos['keypoints_visible']

                input_dict.update({
                    'gt_keypoints': keypoints.astype(np.float32).reshape(-1, self.num_keypoints, 3),
                    'gt_keypoints_visible': keypoints_visible.astype(np.int32).reshape(-1, self.num_keypoints)
                })

        data_dict = self.prepare_data(data_dict=input_dict)

        if data_dict.get('gt_boxes', None) is not None and len(data_dict['gt_boxes']) == 0:
            data_dict['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
            if 'gt_keypoints' in data_dict:
                data_dict['gt_keypoints'] = np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
                data_dict['gt_keypoints_visible'] = np.zeros((0, self.num_keypoints), dtype=np.int32)

        return data_dict


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
    dataset = CustomDataset(
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