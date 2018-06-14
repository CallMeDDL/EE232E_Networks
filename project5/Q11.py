
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
    points[i,0]=x[i]
    points[i,1]=y[i]

# print(points[0,0],points[0,1])
# points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
tri = Delaunay(points)

plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()

