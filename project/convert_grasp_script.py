# This script converts 2D grasps to 3D grasps

import pprint
import numpy as np
import math

from pypcd import pypcd

cloud = pypcd.PointCloud.from_path('resources/pcd0144.txt')

arr = np.array(cloud.pc_data)
arr = [[a,b,c,d,e] for (a,b,c,d,e) in arr]
arr = np.array(arr)
arr = np.delete(arr, [3], axis=1)

# fetch box point cloud
box_model = np.load("resources/box_model.npy")

# join box point cloud and 2D indices
temp1 = box_model
temp2 = arr[:,3:4]
arr = np.hstack((temp1, temp2))

dict = {} 
for i in range(len(arr)):
	index = arr[i][3]
	row = int(math.floor(index/640) + 1)
	col = int((index % 640) + 1)
	dict[(col,row)] = (arr[i][0], arr[i][1], arr[i][2])

# input: first three coordinates of model output 
# returns: 6 coordinates {x, y, z, roll, pitch, yaw} of gripper
# note theta is negated due to differing conventions
# note we have to move down the z coordinate by 0.03 to allow grasping
def convert_grasp(x,y,theta): 
	return (dict[(x,y)][0], dict[(x,y)][1], dict[(x,y)][2] - 0.03, -theta, 5/6*np.pi, 0)

# Example
# CNN returns {236, 227, -0.14, 29, 58} for 2D grasp rectangle
print("3D grasp: ", convert_grasp(236, 227, -0.14))
