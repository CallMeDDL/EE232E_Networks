
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from io import open


f1name = 'coordinate_X.txt'
f2name = 'coordinate_Y.txt'
f3name = 'tour.txt'

x=np.loadtxt(f1name)
y=np.loadtxt(f2name)
tour=[]
file=open(f3name,'r',encoding='UTF-8')
counter=0
for line in file.readlines():
    tour.append(int(line))
print(tour[0])

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

# Plot the approximate tour
Num=len(tour)
# print(tour)
Points=np.zeros(shape=(Num,2))
for i in range(0,Num):
    current=tour[i]-1
    Points[i,0]=x[current]
    Points[i,1]=y[current]

plt.figure(figsize=(13,10))
plt.plot(Points[:,0],Points[:,1])
plt.show()