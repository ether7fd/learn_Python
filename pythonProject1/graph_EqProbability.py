import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

kensa_A = np.loadtxt('learn_Python/pythonProject1/2val_A.csv', delimiter=',', dtype='float')
kensa_B = np.loadtxt('learn_Python/pythonProject1/2val_B.csv', delimiter=',', dtype='float')

def func1(x, y):
    return x**2 + y**2

def COEFFICIENT(data2dim):
    Cov = np.cov(data2dim, rowvar=0, bias=0)
    r = Cov[0][1] / (np.sqrt(Cov[0][0] * Cov[1][1]))
    return r 

def DENSITY_PROPA(avg, var, data):
    dp = 1 / math.sqrt(2 * math.pi * var) * np.exp(-(data-avg)**2 / (2*var))
    return dp

def NORMALDIST_2VAL(avgA, avgB, varA, varB, coffic, dataA, dataB):
    C = 1 / (2 * math.pi * math.sqrt(varA * varB))
    d = -1/(2 * (1-coffic**2)) * (((dataA - avgA)**2) / varA + ((dataB - avgB)**2) / varB - (2*coffic*(dataA - avgA) * (dataB - avgB)) / (math.sqrt(varA * varB)))
    return C * np.exp(d)

xa = 0.6
xb = 0.5
x = np.array([[xa, xa], [xb, xb]])

average = np.array([kensa_A.mean(axis=0), kensa_B.mean(axis=0)])
variance = np.array([kensa_A.var(axis=0, ddof=1), kensa_B.var(axis=0, ddof=1)])

group1 = np.transpose([[row[0] for row in kensa_A], [row[0] for row in kensa_B]])
group2 = np.transpose([[row[1] for row in kensa_A], [row[1] for row in kensa_B]])

#group1A = np.array([[0] for row in group1])

#print(group1[:, [0]])
print('(group1)検査AとBの相関係数:', COEFFICIENT(group1))
xA = group1[:, [0]]
dpA = DENSITY_PROPA(average[0][0], variance[0][0], group1[:, [0]])

xB = group1[:, [1]]
dpB = DENSITY_PROPA(average[1][0], variance[1][0], group1[:, [1]])

normal_2val = NORMALDIST_2VAL(average[0][0], average[1][0], variance[0][0], variance[1][0], COEFFICIENT(group1), group1[:, [0]], group1[:, [1]])

#plt.scatter(xA, dpA, color='b', label='A')
#plt.scatter(xB, dpB, color='r', label='B')
#plt.savefig("learn_Python/pythonProject1/img/density_propability.png")
#plt.show()

X = np.arange(0, 1.0, 0.01)
Y = np.arange(0, 1.0, 0.01)

#Z = NORMALDIST_2VAL(average[0][0], average[1][0], variance[0][0], variance[1][0], COEFFICIENT(group1), X, Y)
Z = func1(X, Y)

print(X,Y,Z)

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.plot(X, Y, Z, marker="o", linestyle='None')
plt.savefig("learn_Python/pythonProject1/img/3Dplot_test.png")
plt.show()
