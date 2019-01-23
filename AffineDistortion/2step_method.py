# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:34:37 2018

@author: user
"""

import cv2
import numpy as np
import os

def Data_Initialize(case):
    if(case == 'a'):
        img = cv2.imread('1.jpg')
        #points on parallel lines
        p = np.matrix([1029,693,1],dtype = float)
        q = np.matrix([759,1257,1],dtype = float)
        r = np.matrix([1548,615,1],dtype = float)
        s = np.matrix([1417,1210,1],dtype = float)
        pts_pll = np.array([p,q,r,s])
        #input image corner pixel coordinates
        p_cnr = np.matrix([0,0,1], dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1], dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1], dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1], dtype = float)
        cnrs = np.array([p_cnr,q_cnr,r_cnr,s_cnr])
        #points on orthogonal lines
        p_orth = np.matrix([946,1243,1], dtype = float)
        q_orth = np.matrix([894,1374,1], dtype = float)
        r_orth = np.matrix([1095,1231,1], dtype = float)
        s_orth = np.matrix([1050,1363,1], dtype = float)
        pts_orth = np.array([p_orth,q_orth,r_orth,s_orth])
    elif (case == 'b'):
        img = cv2.imread('2.jpg') 
        p = np.matrix([71,246,1],dtype = float)
        q = np.matrix([82,326,1],dtype = float)
        r = np.matrix([269,245,1],dtype = float)
        s = np.matrix([265,323,1],dtype = float)
        pts_pll = np.array([p,q,r,s])
        p_cnr = np.matrix([0,0,1],dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1],dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1],dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1],dtype = float)
        cnrs = np.array([p_cnr,q_cnr,s_cnr,r_cnr])
        p_orth = np.matrix([71,246,1], dtype = float)
        q_orth = np.matrix([82,326,1], dtype = float)
        r_orth = np.matrix([149,246,1], dtype = float)
        s_orth = np.matrix([154,323,1], dtype = float)
        pts_orth = np.array([p_orth,q_orth,r_orth,s_orth])
    elif(case == 'c'):
        img = cv2.imread('3.jpg')
        p = np.matrix([136,618,1],dtype = float)
        q = np.matrix([252,1092,1],dtype = float)
        r = np.matrix([396,613,1],dtype = float)
        s = np.matrix([462,1098,1],dtype = float)
        pts_pll = np.array([p,q,r,s])
        p_cnr = np.matrix([0,0,1],dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1],dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1],dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1],dtype = float)
        cnrs = np.array([p_cnr,q_cnr,s_cnr,r_cnr])
        p_orth = np.matrix([322,803,1], dtype = float)
        q_orth = np.matrix([338,887,1], dtype = float)
        r_orth = np.matrix([420,804,1], dtype = float)
        s_orth = np.matrix([432,888,1], dtype = float)
        pts_orth = np.array([p_orth,q_orth,r_orth,s_orth])
    else:
        img = cv2.imread('4.jpg')
        p = np.matrix([14,23,1],dtype = float)
        q = np.matrix([30,146,1],dtype = float)
        r = np.matrix([243,24,1],dtype = float)
        s = np.matrix([223,146,1],dtype = float)
        pts_pll = np.array([p,q,r,s])
        p_cnr = np.matrix([0,0,1],dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1],dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1],dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1],dtype = float)
        cnrs = np.array([p_cnr,q_cnr,s_cnr,r_cnr])
        p_orth = np.matrix([41,40,1], dtype = float)
        q_orth = np.matrix([50,134,1], dtype = float)
        r_orth = np.matrix([158,39,1], dtype = float)
        s_orth = np.matrix([147,135,1], dtype = float)
        pts_orth = np.array([p_orth,q_orth,r_orth,s_orth])  
    return img, pts_pll, cnrs, pts_orth    
    
    
def Homography_VL(pts):
    line1 = np.cross(pts[0], pts[2])
    line2 = np.cross(pts[1], pts[3])
    line3 = np.cross(pts[0], pts[1])
    line4 = np.cross(pts[2], pts[3])
    
    idealpt1 = np.cross(line1, line2)
    idealpt2 = np.cross(line3, line4)
    
    vline = np.cross(idealpt1, idealpt2) 
    vline = vline/vline[0,2]
    
    H = np.zeros((3,3))
    H[0,:] = [1,0,0]
    H[1,:] = [0,1,0]
    H[2,:] = [vline[0,0],vline[0,1],vline[0,2]]
    return H

def Homography_Affine(pts):
    #calculate pair of orthogonal lines
    l1 = np.cross(pts[0].T, pts[1].T) #PQ
    m1 = np.cross(pts[1].T, pts[3].T) #QS
    l2 = np.cross(pts[3].T, pts[0].T) #SP
    m2 = np.cross(pts[2].T, pts[1].T) #RQ
    #Least squares estimation to find s11, s12
    X1 = np.array([l1[0,0]*m1[0,0],l1[0,0]*m1[0,1] + l1[0,1]*m1[0,0]])
    X2 = np.array([l2[0,0]*m2[0,0],l2[0,0]*m2[0,1] + l2[0,1]*m2[0,0]])
    X = np.vstack((X1,X2))
    y = np.matrix([-m1[0,1]*l1[0,1], -m2[0,1]*l2[0,1]])
    XXT_inv = np.linalg.inv(X.T.dot(X))
    B = XXT_inv.dot(X.T).dot(y.T)
    S = np.zeros((2,2))
    S[0,:] = [B[0],B[1]]
    S[1,:] = [B[1], 1]
   
    #Calculate A from SVD
    U,D,V = np.linalg.svd(S) 
    D = np.diag(D)
    A = V.dot(np.sqrt(D)).dot(V.T)
    #Generate affine removal homography
    H = np.zeros((3,3))
    H[0:2,0:2] = A
    H[2,2] = 1   
    return H
    
def PointProjection(H, pts):
    #Apply homography on points
    p_w = np.matmul(H , pts[0].T)
    p_w = p_w / p_w[2]
    p_w[0] = int(p_w[0])
    p_w[1] = int(p_w[1])
    q_w = np.matmul(H , pts[1].T)
    q_w = q_w / q_w[2]
    q_w[0] = int(q_w[0])
    q_w[1] = int(q_w[1])
    r_w = np.matmul(H , pts[2].T)
    r_w = r_w / r_w[2]
    r_w[0] = int(r_w[0])
    r_w[1] = int(r_w[1])
    s_w = np.matmul(H , pts[3].T)
    s_w = s_w / s_w[2]
    s_w[0] = int(s_w[0])
    s_w[1] = int(s_w[1]) 
    pts_world = np.array([p_w,q_w,r_w,s_w])
    
    dim_x = np.amax([p_w[1], q_w[1], r_w[1], s_w[1]])
    dim_y = np.amax([p_w[0], q_w[0], r_w[0], s_w[0]])
    dimensions = np.array([dim_x,dim_y])
    origin_x = np.amin([p_w[1], q_w[1], r_w[1], s_w[1]])
    origin_y = np.amin([p_w[0], q_w[0], r_w[0], s_w[0]])
    origin = np.array([origin_x,origin_y])
    return pts_world, dimensions, origin

image, pts_pll, img_cnrs, pts_orth = Data_Initialize('d')
H_vl = Homography_VL(pts_pll)

pts_Hvl_w, dim_w, origin_w = PointProjection(H_vl, img_cnrs)

#Generate image without project distortion only
"""width = dim_w[0] - origin_w[0]
height = dim_w[1] - origin_w[1]
scale_y =  image.shape[1]/height
scale_x = image.shape[0]/width
w_new = int(width)
h_new = int(height)

new_img1 = np.zeros((h_new,w_new,3))
print(new_img1.shape)
for rows in range(0,new_img1.shape[0]-1):
    for cols in range(0,new_img1.shape[1]-1):
    
        pt_y = (rows) + origin_w[1] - 1
        pt_x = (cols) + origin_w[0] - 1
        
        new_pt = np.matrix([pt_y,pt_x,1])
        map_pt = np.matmul(np.linalg.inv(H_vl), new_pt.T)
        #We care only about ratio
        map_pt = map_pt/map_pt[2]
        
        #print(mapping_pt)
        if map_pt[0]>0 and map_pt[1]>0 and map_pt[0]<image.shape[0]-1 and map_pt[1]<image.shape[1]-1:
            new_img1[rows,cols] = image[int(map_pt[0]),int(map_pt[1])]
            
path = 'C:/Users/user/Desktop/ECE661/HW3/2step'            
cv2.imwrite(os.path.join(path,'2step_ArtRmProj.jpg'), new_img1)
cv2.destroyAllWindows()"""

#Remove projective distortion on points along orthogonal lines
pts_Hvl, dim, origin = PointProjection(H_vl, pts_orth)
#Find affine homography
H_aff = Homography_Affine(pts_Hvl)
Hinv_aff = np.linalg.inv(H_aff)
H = Hinv_aff.dot(H_vl)
#Generate undistorted output image shape
pts_Aff, dim_Aff, origin_Aff = PointProjection(H, img_cnrs)

width = dim_Aff[0] - origin_Aff[0]
height = dim_Aff[1] - origin_Aff[1]
w_new = int(width)
h_new = int(height)
new_img = np.zeros((h_new,w_new,3))

for rows in range(0,new_img.shape[0]-1):
    for cols in range(0,new_img.shape[1]-1):
        pt_y = (rows) + origin_Aff[1] - 1
        pt_x = (cols) + origin_Aff[0] - 1
        new_pt = np.matrix([pt_y,pt_x,1])
        map_pt = np.matmul(np.linalg.inv(H), new_pt.T)
        #care only about ratio
        map_pt = map_pt/map_pt[2]
        #print(mapping_pt)
        if map_pt[0]>0 and map_pt[1]>0 and map_pt[0]<image.shape[0]-1 and map_pt[1]<image.shape[1]-1:
            new_img[rows,cols] = image[int(map_pt[0]),int(map_pt[1])]
            
path = 'C:/Users/user/Desktop/ECE661/HW3/2step'            
cv2.imwrite(os.path.join(path,'2step_Artfinal2.jpg'), new_img)
cv2.destroyAllWindows()
    