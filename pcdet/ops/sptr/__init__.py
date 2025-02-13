try:
    from pcdet.ops.sptr.sptr_cuda import *
except ImportError:
    from .sptr_cuda import *

# 导入其他模块
try:
    from pcdet.ops.sptr.sptr.functional import *
    from pcdet.ops.sptr.sptr.modules import *
    from pcdet.ops.sptr.sptr.position_embedding import *
    from pcdet.ops.sptr.sptr.utils import *
except ImportError:
    from .sptr.functional import *
    from .sptr.modules import *
    from .sptr.position_embedding import *
    from .sptr.utils import *

__all__ = ['sptr_cuda']