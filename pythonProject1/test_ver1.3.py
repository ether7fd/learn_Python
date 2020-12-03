import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

kensa_A = np.loadtxt('./2val_A.csv', delimiter=',', dtype='float')
kensa_B = np.loadtxt('./2val_B.csv', delimiter=',', dtype='float')

def MAHARANO(x, dataA, dataB):
    average = np.array([dataA.mean(axis=0), dataB.mean(axis=0)])
    variance = np.array([dataA.var(axis=0, ddof=1), dataB.var(axis=0, ddof=1)])
    n = np.abs(x - average)
    d = n / np.sqrt(variance)
    return d

xa = 0.6
xb = 0.5
x = np.array([[xa, xa], [xb, xb]])
print("マハラノビスの汎距離：")
print(MAHARANO(x, kensa_A, kensa_B))

group1 = np.transpose([[row[0] for row in kensa_A], [row[0] for row in kensa_B]])
group2 = np.transpose([[row[1] for row in kensa_A], [row[1] for row in kensa_B]])

plt.scatter(group1[:,0], group1[:,1], color='b', label='A')
plt.scatter(group2[:,0], group2[:,1], color='r', label='B')
plt.savefig("./img/hoge.png")
plt.show()

