import numpy as np
import math

f = np.loadtxt('C:/Users/Ether7fd/PycharmProjects/pythonProject1/test_py.csv', delimiter=',', dtype='float')
np.save('test_py.npy', f)
test = np.load('test_py.npy')

heikin = test.mean(axis=0)
print("平均1:", heikin[0])
print("平均2:", heikin[1])

bunsan = test.var(axis=0,ddof=1)
print("分散1:", bunsan[0])
print("分散2:", bunsan[1])