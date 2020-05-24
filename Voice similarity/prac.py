import numpy as np

a = list([[1, 2, 3, 4], [5, 6, 7, 8]])
b = list([[5, 6, 7, 8], [1, 2, 3, 4]])

np.save("a.npy", arr=a)
np.save("b.npy", arr=b)

np.savez("ab.npz", a=a, b=b)

c = np.load("a.npy")
d = np.load("b.npy")
e = np.load("ab.npz")

print("c: ", c)
print("d: ", d)
print("e: ", e['a'])
print("f: ", e['b'])
