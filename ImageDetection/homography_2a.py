# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:18:55 2018

@author: user
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
img_src = cv2.imread('neymar.jpg')

#prepare map to fill in transformed image
shape_a = [img_a.shape[0],img_a.shape[1],3]
frame_a = np.zeros(shape_a, dtype='uint8')
pts_a = np.array([[2616,1936],[2686,1098],[1250,1090],[1284,2102]], np.int32)
cv2.fillPoly(frame_a,[pts_a],(255,255,255))

shape_b = [img_b.shape[0],img_b.shape[1],3]
frame_b = np.zeros(shape_b, dtype='uint8')
pts_b = np.array([[882,1012],[2534,1002],[2452,1914],[936,1918]], np.int32)
cv2.fillPoly(frame_b,[pts_b],(255,255,255))

shape_c = [img_c.shape[0],img_c.shape[1],3]
frame_c = np.zeros(shape_c, dtype='uint8')
pts_c = np.array([[861,1191],[920,1910],[2053,2034],[2082,1120]], np.int32)
cv2.fillPoly(frame_c,[pts_c],(255,255,255))

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

p_src = np.matrix([0,0,1],dtype = float)
q_src = np.matrix([0,1280,1],dtype = float)
r_src = np.matrix([720,0,1],dtype = float)
s_src = np.matrix([720,1280,1],dtype = float)
pts_src = np.array([p_src,q_src,r_src,s_src])

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
dlt = DLT(pts_c,pts_src)
h = Homography(dlt)
for rows in range(0,img_c.shape[0]-1):
    for cols in range(0,img_c.shape[1]-1):
        if frame_c[rows,cols,1] > 0:
            pt = np.matrix([rows,cols,1])
            new_pt = np.matmul(h, pt.T)
            new_pt = new_pt/new_pt[2]     
            if new_pt[0]>0 and new_pt[1]>0 and new_pt[0]<img_src.shape[0]-1 and new_pt[1]<img_src.shape[1]-1:
               img_c[rows,cols] = img_src[int(new_pt[0]),int(new_pt[1])]
                   
cv2.imwrite('Output3.jpg', img_c)
cv2.destroyAllWindows()
    




