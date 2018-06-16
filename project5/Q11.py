
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from io import open


f1name = 'coordinate_X.txt'
f2name = 'coordinate_Y.txt'

x=np.loadtxt(f1name)
y=np.loadtxt(f2name)

Total=len(x)
points=np.zeros(shape=(Total,2))

for i in range(0,Total):
    points[i,0] = x[i]
    points[i,1] = y[i]

tri = Delaunay(points)

"""

plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
"""

print(len(tri.simplices))
print(Total)

plt.triplot(points[:,0], points[:,1])
plt.show()

