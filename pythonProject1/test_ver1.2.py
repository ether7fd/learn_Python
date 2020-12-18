import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

kensa_A = np.loadtxt("./2val_A.csv", delimiter=',', dtype='float')
kensa_B = np.loadtxt('./2val_B.csv', delimiter=',', dtype='float')

def MAHARANO(x, average, variance):
    n = math.fabs(x - average)
    d = n / math.sqrt(variance)
    return d

# x = float(input('input data:'))
xa = 0.6
xb = 0.5

heikina = kensa_A.mean(axis=0)
heikinb = kensa_B.mean(axis=0)

bunsana = kensa_A.var(axis=0, ddof=1)
bunsanb = kensa_B.var(axis=0, ddof=1)

print("マハラノビスの汎距離1(検査A):", MAHARANO(xa, heikina[0], bunsana[0]))
print("マハラノビスの汎距離2(検査A):", MAHARANO(xa, heikina[1], bunsana[1]))
print("マハラノビスの汎距離1(検査B):", MAHARANO(xb, heikinb[0], bunsanb[0]))
print("マハラノビスの汎距離2(検査B):", MAHARANO(xb, heikinb[1], bunsanb[1]))

x = [100, 200, 300, 400, 500, 600]
y = [10, 20, 30, 50, 80, 130]

plt.scatter(x, y)
plt.savefig("./img/hoge.png") # この行を追記
plt.show()

