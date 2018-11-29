import numpy as np

a = np.array([1,2])

b = np.array([2,1])

dot = 0

for (e, f in zip(a,b)):
    dot += e*f

print(dot)