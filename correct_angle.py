import math
import os, json
import openpyxl
import numpy as np
from scipy.optimize import leastsq
import numpy.lib.recfunctions as recfunctions
import matplotlib.pyplot as plt

data_raw = np.genfromtxt('D:/Celegans-locomotion/data/06/body.dat', delimiter=' ', dtype=None)
data_raw = data_raw[:, 1:]
n_tstep = data_raw.shape[0]
n_seg = data_raw.shape[1] // 3
data_raw = np.reshape(data_raw, (
n_tstep, n_seg, len(np.dtype([('x', 'f8'), ('y', 'f8'), ('phi', 'f8')]))))  # type: ignore
data_raw = recfunctions.unstructured_to_structured(
    data_raw,
    dtype=np.dtype([('x', 'f8'), ('y', 'f8'), ('phi', 'f8')]),
)

def func(p, x):
    k, b = p
    return k * x + b

def error(p, x, y):
    return func(p, x) - y

p0 = [1, 1]

Xi = data_raw['x'][200:1000, 0]
Yi = data_raw['y'][200:1000, 0]
Para = leastsq(error, p0, args=(Xi, Yi))

k, b = Para[0]
print("k=", k, "b=", b, "phi=", 1.57-(3.1416+math.atan(k)-3.1416/2))
print("cost：" + str(Para[1]))
print("y=" + str(round(k, 2)) + "x+" + str(round(b, 2)))



plt.figure(figsize=(8, 6))
plt.scatter(data_raw['x'][:, 0], data_raw['y'][:, 0], color="green",  linewidth=1)


x = np.array([Xi[0], Xi[-1]])
y = k * x + b
plt.plot(x, y, color="red",  linewidth=2)
plt.title('y={}+{}x'.format(b,k))
plt.legend(loc='lower right')
plt.show()

