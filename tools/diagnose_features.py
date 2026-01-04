"""
特征诊断脚本：用于分析模型在推理时的特征分布
运行方式：
python tools/diagnose_features.py --cfg_file CONFIG --ckpt CHECKPOINT
"""

import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from eval_utils import eval_utils

# 导入诊断工具
from pcdet.models.model_utils.feature_diagnostics import (
    get_diagnostics,
    enable_diagnostics,
    disable_diagnostics,
    clear_diagnostics,
    print_diagnostics
)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples to diagnose')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np_random_seed = 1024
    torch.manual_seed(np_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return args, cfg


def eval_single_ckpt_with_diagnostics(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # 启用诊断
    enable_diagnostics()
    clear_diagnostics()

    # 使用原始的evaluation函数，但只评估少量样本
    logger.info(f'*************** 开始诊断 (前{args.num_samples}个样本) ***************')

    # 临时修改数据集大小
    original_dataset = test_loader.dataset
    original_len = len(original_dataset)

    # 只处理前N个样本
    model.eval()

    diagnostics = get_diagnostics()

    with torch.no_grad():
        for i, batch_dict in enumerate(test_loader):
            if i >= args.num_samples:
                break

            logger.info(f'处理样本 {i+1}/{args.num_samples}')

            # 将数据加载到GPU
            load_data_to_gpu(batch_dict)

            # 前向传播（会自动记录诊断信息）
            pred_dicts, ret_dict = model(batch_dict)

            # 每个样本打印一次诊断
            if i == 0:
                print_diagnostics(title=f"样本 {i+1} 特征诊断报告")

    # 打印综合诊断报告
    logger.info('\n' + '='*100)
    logger.info('综合诊断报告')
    logger.info('='*100)
    print_diagnostics(title="所有样本的特征统计")

    # 禁用诊断
    disable_diagnostics()

    logger.info('诊断完成')


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def main():
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.eval_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_all:
        assert os.path.exists(args.ckpt_dir)
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_diagnose_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************开始诊断**********************')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # 开始诊断
    eval_single_ckpt_with_diagnostics(model, test_loader, args, eval_output_dir, logger, epoch_id=0, dist_test=dist_test)


if __name__ == '__main__':
    import numpy as np
    import re
    main()
