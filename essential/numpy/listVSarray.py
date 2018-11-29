import numpy as np

L = [1,2,3]

A = np.array([1,2,3])

# print(A)

for e in A:
    print(e)

L.append(4)
L = L + [5]
print(L)

L2 = []

for e in L:
    L2.append(e+e)

print(L2)

print(A+A) # vector edition

print(2*A)

print(2*L)

# numpy functions are working element-wise

A**2

np.sqrt(A)

np.log(A)

np.exp(A)