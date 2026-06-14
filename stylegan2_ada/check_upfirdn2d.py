import sys
import warnings
import traceback

sys.path.insert(
    0,
    "/hpc_stor03/sjtu_home/siru.ge/DeepLearningHomework/stylegan2-ada-pytorch",
)

import torch
from torch_utils.ops import upfirdn2d
from torch_utils.ops.upfirdn2d import _init

print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")

try:
    import ninja

    print(f"ninja: {ninja.__version__}")
except Exception as e:
    print(f"ninja: MISSING ({e})")

print("\n--- forcing _init() (triggers JIT compile) ---")
ok = False
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        ok = _init()
    except Exception as e:
        print(f"_init raised: {e}")
        traceback.print_exc()

    for warning in w:
        print(f"WARNING: {warning.message}")

print(f"_init returned: {ok}, _plugin loaded: {upfirdn2d._plugin is not None}")

print("\n--- real cuda call ---")
try:
    x = torch.randn(1, 3, 8, 8, device="cuda")
    f = torch.ones(4, device="cuda")
    y = upfirdn2d.upfirdn2d(x, f, up=2, padding=2, impl="cuda")
    print(f"OK, out shape: {y.shape}, _plugin: {upfirdn2d._plugin is not None}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
