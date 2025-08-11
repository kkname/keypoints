import argparse
import glob
from pathlib import Path

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
import time

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

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Continuous Playback Demo of OpenPCDet-------------------------')

    # 1. 加载所有数据帧
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of frames to play: \t{len(demo_dataset)}')

    # 2. 构建并加载模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # --- 关键修改：初始化一个持久的可视化窗口 ---
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='VoxelKP Continuous Playback', width=1920, height=1080)
    vis.get_render_option().point_size = 1.5
    vis.get_render_option().background_color = np.zeros(3)
    # ---------------------------------------------

    ctr = vis.get_view_control()
    ctr.set_front([-0.7, -0.7, -0.2])  # 设置相机朝向 (从斜后上方看)
    ctr.set_lookat([5, 0, 0])  # 设置相机对准的点 (场景中前方5米处)
    ctr.set_up([0, 0, 1])  # 设置相机的“上”方向 (Z轴朝上)
    ctr.set_zoom(0.1)  # 设置初始缩放级别

    # 添加一个只创建一次的坐标系
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    # 准备一个点云对象，后续只更新它的点
    pcd = open3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # 获取视图控制器，可以设置初始视角
    ctr = vis.get_view_control()
    try:
        parameters = open3d.io.read_pinhole_camera_parameters("tools/visual_utils/camera_pose_2.json")
        ctr.convert_from_pinhole_camera_parameters(parameters)
    except Exception:
        pass  # 如果相机参数文件不存在，则使用默认视角

    # --- 关键修改：逐帧处理和更新循环 ---
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing frame index: \t{idx}')

            # 准备数据并进行推理
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # 获取点云和预测结果
            points = data_dict['points'][:, 1:]
            pred_boxes = pred_dicts[0].get('pred_boxes', None)
            pred_kps = pred_dicts[0].get('pred_kps', None)

            # --- 更新可视化内容 ---
            # 1. 更新点云
            pcd.points = open3d.utility.Vector3dVector(points[:, :3].cpu().numpy())
            vis.update_geometry(pcd)

            # 2. 清除上一帧的几何体 (除了点云和坐标系)
            vis.clear_geometries()  # 这会清除所有东西，所以我们再把点云和坐标系加回来
            vis.add_geometry(pcd, reset_bounding_box=False)
            vis.add_geometry(axis_pcd, reset_bounding_box=False)

            # 3. 添加新一帧的预测框和关键点
            #    我们直接调用 open3d_vis_utils.py 中的 obtain_all_geometry 函数来创建模型
            #    注意我们只传入了预测结果，并且不绘制原点(因为已经有了)
            new_geometries = V.obtain_all_geometry(
                points=np.zeros((0, 3)),  # 传入空点云，因为我们已经单独处理了
                ref_boxes=pred_boxes,
                ref_keypoints=pred_kps,
                draw_origin=False
            )
            for geom in new_geometries:
                vis.add_geometry(geom, reset_bounding_box=False)
            # ---------------------

            # 更新渲染器并处理事件
            vis.poll_events()
            vis.update_renderer()

            # 控制播放速度 (例如，每帧之间暂停0.1秒)
            time.sleep(0.1)

    logger.info('Playback done.')
    # 播放结束后，销毁窗口
    vis.destroy_window()


if __name__ == '__main__':
    main()
