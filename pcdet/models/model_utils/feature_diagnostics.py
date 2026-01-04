"""
特征诊断工具：用于追踪特征在模型中的变化
"""
import torch
import torch.nn as nn
from ...utils.spconv_utils import spconv


class FeatureDiagnostics:
    """追踪和记录特征统计信息"""

    def __init__(self):
        self.logs = []
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def clear(self):
        self.logs = []

    @torch.no_grad()
    def log_dense_tensor(self, tensor, name, prefix=""):
        """记录密集张量的统计信息"""
        if not self.enabled:
            return

        stats = {
            'name': f"{prefix}{name}",
            'shape': tuple(tensor.shape),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'abs_mean': tensor.abs().mean().item(),
        }

        # 计算通道级别的统计（如果是4D张量）
        if tensor.ndim == 4:  # (B, C, H, W)
            channel_means = tensor.mean(dim=[0, 2, 3])  # (C,)
            stats['channel_mean_range'] = (channel_means.min().item(), channel_means.max().item())
            stats['channel_std'] = channel_means.std().item()

        self.logs.append(stats)
        return stats

    @torch.no_grad()
    def log_sparse_tensor(self, sparse_tensor, name, prefix=""):
        """记录稀疏张量的统计信息"""
        if not self.enabled:
            return

        features = sparse_tensor.features  # (N, C)
        stats = {
            'name': f"{prefix}{name}",
            'type': 'SparseConvTensor',
            'num_active_voxels': features.shape[0],
            'num_channels': features.shape[1],
            'mean': features.mean().item(),
            'std': features.std().item(),
            'min': features.min().item(),
            'max': features.max().item(),
            'abs_mean': features.abs().mean().item(),
        }

        # 通道级别的统计
        channel_means = features.mean(dim=0)  # (C,)
        stats['channel_mean_range'] = (channel_means.min().item(), channel_means.max().item())
        stats['channel_std'] = channel_means.std().item()

        self.logs.append(stats)
        return stats

    def print_summary(self, title="Feature Diagnostics Summary"):
        """打印诊断摘要"""
        if not self.logs:
            print(f"\n{'='*80}\n{title}\n{'='*80}")
            print("No logs recorded.")
            return

        print(f"\n{'='*80}\n{title}\n{'='*80}")
        for i, log in enumerate(self.logs):
            print(f"\n[{i+1}] {log['name']}")
            if 'type' in log and log['type'] == 'SparseConvTensor':
                print(f"  Type: SparseConvTensor")
                print(f"  Active voxels: {log['num_active_voxels']}, Channels: {log['num_channels']}")
            else:
                print(f"  Shape: {log['shape']}")

            print(f"  Range: [{log['min']:.4f}, {log['max']:.4f}]")
            print(f"  Mean: {log['mean']:.4f}, Std: {log['std']:.4f}")
            print(f"  Abs Mean: {log['abs_mean']:.4f}")

            if 'channel_mean_range' in log:
                print(f"  Channel mean range: [{log['channel_mean_range'][0]:.4f}, {log['channel_mean_range'][1]:.4f}]")
                print(f"  Channel std: {log['channel_std']:.4f}")
        print(f"{'='*80}\n")

    def get_comparison(self, name1, name2):
        """对比两个特征的差异"""
        log1 = next((l for l in self.logs if l['name'] == name1), None)
        log2 = next((l for l in self.logs if l['name'] == name2), None)

        if log1 is None or log2 is None:
            return None

        comparison = {
            'mean_diff': abs(log1['mean'] - log2['mean']),
            'std_diff': abs(log1['std'] - log2['std']),
            'range_diff': (abs(log1['min'] - log2['min']), abs(log1['max'] - log2['max'])),
        }
        return comparison


class SeparateHeadWithDiagnostics(nn.Module):
    """带诊断功能的SeparateHead"""

    def __init__(self, original_head, diagnostics, head_name="head_0"):
        super().__init__()
        self.original_head = original_head
        self.diagnostics = diagnostics
        self.head_name = head_name
        self.sep_head_dict = original_head.sep_head_dict

    def forward(self, x):
        # 记录输入特征
        self.diagnostics.log_sparse_tensor(x, "input", prefix=f"{self.head_name}.")

        ret_dict = {}
        for cur_name in self.sep_head_dict:
            # 获取该分支的卷积层序列
            branch = self.original_head.__getattr__(cur_name)

            # 逐层执行并记录
            intermediate = x
            for layer_idx, layer in enumerate(branch):
                intermediate = layer(intermediate)

                # 如果是hm分支，记录每一层
                if 'hm' in cur_name:
                    self.diagnostics.log_sparse_tensor(
                        intermediate,
                        f"{cur_name}_layer{layer_idx}",
                        prefix=f"{self.head_name}."
                    )

            # 提取最终特征
            ret_dict[cur_name] = intermediate.features

            # 记录输出（已经是dense features）
            if 'hm' in cur_name:
                self.diagnostics.log_dense_tensor(
                    intermediate.features.unsqueeze(0),
                    f"{cur_name}_output",
                    prefix=f"{self.head_name}."
                )

        return ret_dict


# 全局诊断实例
global_diagnostics = FeatureDiagnostics()


def enable_diagnostics():
    """启用诊断"""
    global_diagnostics.enable()


def disable_diagnostics():
    """禁用诊断"""
    global_diagnostics.disable()


def clear_diagnostics():
    """清空诊断日志"""
    global_diagnostics.clear()


def print_diagnostics(title="Feature Diagnostics"):
    """打印诊断报告"""
    global_diagnostics.print_summary(title)


def get_diagnostics():
    """获取诊断实例"""
    return global_diagnostics
