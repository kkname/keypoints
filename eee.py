import os
import sys
import torch

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)

try:
    from pcdet.ops.sptr import sptr_cuda
    print("\nsptr_cuda successfully imported!")
    print("\nAvailable functions in sptr_cuda:")
    print(dir(sptr_cuda))
except Exception as e:
    print("Error loading sptr_cuda:", str(e))
    import traceback
    traceback.print_exc()