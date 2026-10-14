import numpy as np
import oqpsk

for sps in (2, 4, 8, 16):
    bits = np.tile([0, 1, 1, 0], 10).astype(np.int8)
    x = oqpsk.modulate(bits, sps=sps)
    expected = (len(bits) // 4) * 32 * sps
    print(f"sps={sps}: actual={len(x)}, expected={expected}, "
          f"extra={len(x)-expected}, finite={np.all(np.isfinite(x))}")
