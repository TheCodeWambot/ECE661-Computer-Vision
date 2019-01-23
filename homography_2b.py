# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:29:18 2018

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:31:10 2018

@author: hank huang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:16:35 2018

@author: hank huang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:16:35 2018

@author: hank huang
"""

import cv2
import numpy as np

#import images
img_a = cv2.imread('sample1.jpg')
img_b = cv2.imread('sample2.jpg')
img_c = cv2.imread('sample3.jpg')

img_buf = np.zeros((img_a.shape[0],img_a.shape[1],3))
#source and destination manual pixel coordinates
p_a = np.matrix([1098,1250,1],dtype = float)
q_a = np.matrix([1098,2686,1],dtype = float)
r_a = np.matrix([2102,1284,1],dtype = float)
s_a = np.matrix([1936,2616,1],dtype = float)
pts_a = np.array([p_a,q_a,r_a,s_a])

p_b = np.matrix([1012,882,1],dtype = float)
q_b = np.matrix([1002,2534,1],dtype = float)
r_b = np.matrix([1918,936,1],dtype = float)
s_b = np.matrix([1914,2452,1],dtype = float)
pts_b = np.array([p_b,q_b,r_b,s_b])

p_c = np.matrix([1191,861,1],dtype = float)
q_c = np.matrix([1120,2082,1],dtype = float)
r_c = np.matrix([1910,920,1],dtype = float)
s_c = np.matrix([2034,2053,1],dtype = float)
pts_c = np.array([p_c,q_c,r_c,s_c])

#direct linear transform
def DLT(dstpt,srcpt):
    A = np.matrix(np.zeros(shape = (8,9)))
    A[0,0:3] = dstpt[0]
    A[0,6:9] = -srcpt[0][0,0]*dstpt[0]
    A[1,3:6] = dstpt[0]
    A[1,6:9] = -srcpt[0][0,1]*dstpt[0]
   
    A[2,0:3] = dstpt[1]
    A[2,6:9] = -srcpt[1][0,0]*dstpt[1]
    A[3,3:6] = dstpt[1]
    A[3,6:9] = -srcpt[1][0,1]*dstpt[1]
    
    A[4,0:3] = dstpt[2]
    A[4,6:9] = -srcpt[2][0,0]*dstpt[2]
    A[5,3:6] = dstpt[2]
    A[5,6:9] = -srcpt[2][0,1]*dstpt[2]
    
    A[6,0:3] = dstpt[3]
    A[6,6:9] = -srcpt[3][0,0]*dstpt[3]
    A[7,3:6] = dstpt[3]
    A[7,6:9] = -srcpt[3][0,1]*dstpt[3]   
    return(A)
    
#find homography with SVD and reshape to 3x3    
def Homography(dlt):
    u,s,vt = np.linalg.svd(dlt) 
    h = vt[8,0:9]
    h_new = np.reshape(h,(3,3))
    return h_new

#project pixels to frame image
dlt_ab = DLT(pts_b,pts_a)
h_ab = Homography(dlt_ab)
dlt_bc = DLT(pts_c,pts_b)
h_bc = Homography(dlt_bc)
h_prod = np.matmul(h_ab,h_bc)

for rows in range(0,img_a.shape[0]-1):
    for cols in range(0,img_a.shape[1]-1):
        
            pt = np.matrix([rows,cols,1])
            new_pt = np.matmul(h_prod, pt.T)
            new_pt = new_pt/new_pt[2]     
            if new_pt[0]>0 and new_pt[1]>0 and new_pt[0]<img_a.shape[0]-1 and new_pt[1]<img_a.shape[1]-1:
               img_buf[rows,cols] = img_a[int(new_pt[0]),int(new_pt[1])]
                   
cv2.imwrite('task2b.jpg', img_buf)
cv2.destroyAllWindows()
    





    




