# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:24:13 2018

@author: hank huang
"""

import cv2
import numpy as np
import os

def Data_Initialize(case):
    if(case == 'a'):
        img = cv2.imread('1.jpg')
        #image plane pixel coordinates
        p_im = np.matrix([760,1258,1],dtype = float)
        q_im = np.matrix([702,1385,1],dtype = float)
        r_im = np.matrix([945,1244,1],dtype = float)
        s_im = np.matrix([893,1374,1],dtype = float)
        pts_im = np.array([p_im,q_im,r_im,s_im])
        #image plane corner coordinates
        p_cnr = np.matrix([0,0,1],dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1], dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1], dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1], dtype = float)
        pts_cnr = np.array([p_cnr,q_cnr,r_cnr,s_cnr])
        p_w = np.matrix([0,0,1],dtype = float)
        #world plane corner coordinates
        q_w = np.matrix([0,60,1],dtype = float)
        r_w = np.matrix([80,0,1],dtype = float)
        s_w = np.matrix([80,60,1],dtype = float)
        pts_w = np.array([p_w,q_w,r_w,s_w])
        
    elif(case == 'b'):
        img = cv2.imread('2.jpg')
        p_im = np.matrix([57,232,1],dtype = float)
        q_im = np.matrix([72,337,1],dtype = float)
        r_im = np.matrix([284,230,1],dtype = float)
        s_im = np.matrix([276,335,1],dtype = float)
        pts_im = np.array([p_im,q_im,s_im,r_im])
        p_cnr = np.matrix([0,0,1],dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1],dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1],dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1],dtype = float)
        pts_cnr = np.array([p_cnr,q_cnr,s_cnr,r_cnr])
        p_w = np.matrix([0,0,1],dtype = float)
        q_w = np.matrix([0,40,1],dtype = float)
        r_w = np.matrix([80,0,1],dtype = float)
        s_w = np.matrix([80,40,1],dtype = float)
        pts_w = np.array([p_w,q_w,s_w,r_w])
    
    return img, pts_im, pts_cnr, pts_w
        
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
    h_array = vt[8,0:9]
    H = np.reshape(h_array,(3,3))
    return H

def PointProjection(H, cnrpts):
    #generate projected world plane corners
    p_wcnr = np.matmul(H , cnrpts[0].T)
    p_wcnr = p_wcnr / p_wcnr[2]
    p_wcnr[0] = int(p_wcnr[0])
    p_wcnr[1] = int(p_wcnr[1])
    q_wcnr = np.matmul(H , cnrpts[1].T)
    q_wcnr = q_wcnr / q_wcnr[2]
    q_wcnr[0] = int(q_wcnr[0])
    q_wcnr[1] = int(q_wcnr[1])
    r_wcnr = np.matmul(H , cnrpts[2].T)
    r_wcnr = r_wcnr / r_wcnr[2]
    r_wcnr[0] = int(r_wcnr[0])
    r_wcnr[1] = int(r_wcnr[1])
    s_wcnr = np.matmul(H , cnrpts[3].T)
    s_wcnr = s_wcnr / s_wcnr[2]
    s_wcnr[0] = int(s_wcnr[0])
    s_wcnr[1] = int(s_wcnr[1])
    
    #define new image dimensions
    dim_x = np.amax([p_wcnr[1], q_wcnr[1], r_wcnr[1], s_wcnr[1]])
    dim_y = np.amax([p_wcnr[0], q_wcnr[0], r_wcnr[0], s_wcnr[0]])
    dimensions = np.array([dim_x,dim_y])
    origin_x = np.amin([p_wcnr[1], q_wcnr[1], r_wcnr[1], s_wcnr[1]])
    origin_y = np.amin([p_wcnr[0], q_wcnr[0], r_wcnr[0], s_wcnr[0]])
    origin = np.array([origin_x,origin_y])
    return (dimensions, origin)
    

image,pts_im,pts_cnr,pts_w = Data_Initialize('b')
dlt = DLT(pts_im,pts_w)
H = Homography(dlt)

#Inverse of H to reverse projection onto image plane
H_inv = np.linalg.inv(H)
dim, origin = PointProjection(H, pts_cnr)
width = dim[0] - origin[0]
height = dim[1] - origin[1]
    
#calculate scale for output image
scale_y =  image.shape[0]/width
scale_x = image.shape[1]/height
if scale_y > scale_x:
    scale = scale_y
else:
    scale = scale_x
w_new = int(width*scale)
h_new = int(height*scale)
new_img = np.zeros((h_new,w_new,3))

for rows in range(0,new_img.shape[0]-1):
    for cols in range(0,new_img.shape[1]-1):
        pt_y = (rows/scale) + origin[1] - 1
        pt_x = (cols/scale) + origin[0]- 1
        new_pt = np.matrix([pt_y,pt_x,1])
        map_pt = np.matmul(H_inv , new_pt.T)
        #We care only about ratio
        map_pt = map_pt/map_pt[2]
        
        if map_pt[0]>0 and map_pt[1]>0 and map_pt[0]<image.shape[0]-1 and map_pt[1]<image.shape[1]-1:
            new_img[rows,cols] = image[int(map_pt[0]),int(map_pt[1])]
                 
path = 'C:/Users/user/Desktop/ECE661/HW3/p2p'            
cv2.imwrite(os.path.join(path,'p2p_building.jpg'), new_img)
cv2.destroyAllWindows()






