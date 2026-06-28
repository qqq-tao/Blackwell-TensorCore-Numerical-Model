import numpy as np

significand = np.uint32(1)
significand = np.bitwise_and(np.uint32(significand), np.uint32(0xFFFFFFFE))
print(significand)
