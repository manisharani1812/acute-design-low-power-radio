import numpy as np
import oqpsk

tests = [
    ("3 bits", [1, 0, 1]),
    ("5 bits", [1, 0, 1, 0, 1]),
    ("invalid bit", [0, 1, 2, 0]),
    ("empty input", []),
]

for name, values in tests:
    try:
        result = oqpsk.modulate(np.array(values, dtype=np.int8), sps=8)
        print(f"{name}: ACCEPTED, output length={len(result)}")
    except Exception as e:
        print(f"{name}: REJECTED, {type(e).__name__}: {e}")
