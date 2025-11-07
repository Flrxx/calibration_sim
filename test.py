import matplotlib.pyplot as plt  # Correct import
from mpl_toolkits.mplot3d import Axes3D 

point_x = 0.1
point_y = 0.2 
point_z = 0.3

print('3D point(x,y,z):',point_x,point_y,point_z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(point_x, point_y, point_z, color='red', s=50)

# Customize the plot (add labels, titles, etc.)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Single Point in 3D')
plt.show()  
