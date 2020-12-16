import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.1)
y = np.arange(0, 1, 0.1)

xx, yy = np.meshgrid(x, y)

z = np.sqrt(xx**2 + yy**2)

print(z)

#plt.scatter(xx, yy)
plt.contourf(xx, yy, z, cmap='Blues')
plt.show()
