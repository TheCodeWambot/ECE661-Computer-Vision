# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:02:01 2018

@author: user
"""
import cv2
import numpy as np
import os

def Data_Initialize(case):
    if(case == 'a'):
        img = cv2.imread('1.jpg')
        #input image corner pixel coordinates
        p_cnr = np.matrix([0,0,1], dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1], dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1], dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1], dtype = float)
        cnrs = np.array([p_cnr,q_cnr,r_cnr,s_cnr])
        #points on orthogonal lines
        p_orth = np.matrix([633,1266,1], dtype = float)
        q_orth = np.matrix([280,2024,1], dtype = float)
        r_orth = np.matrix([1420,1206,1], dtype = float)
        s_orth = np.matrix([1218,2078,1], dtype = float)
        pts_orth = np.array([p_orth,q_orth,r_orth,s_orth])
    elif(case == 'b'):
        img = cv2.imread('2.jpg')
        p_cnr = np.matrix([0,0,1], dtype = float)
        q_cnr = np.matrix([0,img.shape[1],1], dtype = float)
        r_cnr = np.matrix([img.shape[0],0,1], dtype = float)
        s_cnr = np.matrix([img.shape[0],img.shape[1],1], dtype = float)
        cnrs = np.array([p_cnr,q_cnr,r_cnr,s_cnr])
        
        p_orth = np.matrix([71,246,1], dtype = float)
        q_orth = np.matrix([82,326,1], dtype = float)
        r_orth = np.matrix([149,246,1], dtype = float)
        s_orth = np.matrix([154,323,1], dtype = float)
        pts_orth = np.array([p_orth,q_orth,r_orth,s_orth])
    elif(case == 'c'):
        img = cv2.imread('3.jpg')
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
           
    return img, cnrs, pts_orth

def Homography_OneStepAffine(pts): 
    #calculate 5 pairs of orthogonal lines
    l1 = np.cross(pts[0], pts[1]) #PQ
    m1 = np.cross(pts[1], pts[3]) #QS
    l2 = np.cross(pts[1], pts[3]) #QS
    m2 = np.cross(pts[3], pts[2]) #SR
    l3 = np.cross(pts[3], pts[2]) #SR
    m3 = np.cross(pts[2], pts[0]) #RP
    l4 = np.cross(pts[2], pts[0]) #RP
    m4 = np.cross(pts[0], pts[1]) #PQ
    l5 = np.cross(pts[0], pts[3]) #PS
    m5 = np.cross(pts[2], pts[1]) #RQ
    
    #Least square estimation for dual degenerate conic parameters
    X1 = np.array([l1[0,0]*m1[0,0], (l1[0,1]*m1[0,0] + l1[0,0]*m1[0,1])/2, (l1[0,1]*m1[0,1]), (l1[0,2]*m1[0,0]+l1[0,0]*m1[0,2])/2, (l1[0,2]*m1[0,1]+l1[0,1]*m1[0,2])/2 ])
    X2 = np.array([l2[0,0]*m2[0,0], (l2[0,1]*m2[0,0] + l2[0,0]*m2[0,1])/2, (l2[0,1]*m2[0,1]), (l2[0,2]*m2[0,0]+l2[0,0]*m2[0,2])/2, (l2[0,2]*m2[0,1]+l2[0,1]*m2[0,2])/2 ])
    X3 = np.array([l3[0,0]*m3[0,0], (l3[0,1]*m3[0,0] + l3[0,0]*m3[0,1])/2, (l3[0,1]*m3[0,1]), (l3[0,2]*m3[0,0]+l3[0,0]*m3[0,2])/2, (l3[0,2]*m3[0,1]+l3[0,1]*m3[0,2])/2 ])
    X4 = np.array([l4[0,0]*m4[0,0], (l4[0,1]*m4[0,0] + l4[0,0]*m4[0,1])/2, (l4[0,1]*m4[0,1]), (l4[0,2]*m4[0,0]+l4[0,0]*m4[0,2])/2, (l4[0,2]*m4[0,1]+l4[0,1]*m4[0,2])/2 ])
    X5 = np.array([l5[0,0]*m5[0,0], (l5[0,1]*m5[0,0] + l5[0,0]*m5[0,1])/2, (l5[0,1]*m5[0,1]), (l5[0,2]*m5[0,0]+l5[0,0]*m5[0,2])/2, (l5[0,2]*m5[0,1]+l5[0,1]*m5[0,2])/2 ])
    X = np.vstack((X1,X2,X3,X4,X5))  
    y = np.matrix([-m1[0,2]*l1[0,2], -m2[0,2]*l2[0,2], -m3[0,2]*l3[0,2], -m4[0,2]*l4[0,2], -m5[0,2]*l5[0,2]],dtype = float) 
    
    XXT_inv = np.linalg.inv(X.T.dot(X))
    param = XXT_inv.dot(X.T).dot(y.T)

    #image of dual degenerate conic
    iDDC = np.zeros((3,3))
    iDDC[0,:] = [param[0], param[1]/2, param[3]/2]
    iDDC[1,:] = [param[1]/2, param[2], param[4]/2]
    iDDC[2,:] = [param[3]/2, param[4]/2, 1]
    #normalize conic
    iDDC = iDDC / iDDC.max()
    #find A and v in homography matrix
    S = np.zeros((2,2))
    S[0,:] = [iDDC[0,0],iDDC[0,1]]
    S[1,:] = [iDDC[1,0],iDDC[1,1]]
    U,D,V = np.linalg.svd(S)
    D = np.diag(D)
    A = V.dot(np.sqrt(D)).dot(V.T)
    v = np.matrix([iDDC[2,0],iDDC[2,1]]) * np.linalg.inv(A.T)
    
    H = np.zeros((3,3))
    H[0:2,0:2] = A
    H[2,0:2] = v
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

image, cnrs, pts_orth = Data_Initialize('d')
H = Homography_OneStepAffine(pts_orth)
H_inv = np.linalg.inv(H)
pts_new, dim, origin = PointProjection(H_inv, cnrs)
width = dim[0] - origin[0]
height = dim[1] - origin[1]
w_new = int(width)
h_new = int(height)
new_img = np.zeros((h_new,w_new,3))

for rows in range(0,new_img.shape[0]-1):
    for cols in range(0,new_img.shape[1]-1):
        pt_y = (rows) + origin[1] - 1
        pt_x = (cols) + origin[0] - 1
        new_pt = np.matrix([pt_y,pt_x,1])
        map_pt = np.matmul(H, new_pt.T)
        #We care only about ratio
        map_pt = map_pt/map_pt[2]
        #print(mapping_pt)
        if map_pt[0]>0 and map_pt[1]>0 and map_pt[0]<image.shape[0]-1 and map_pt[1]<image.shape[1]-1:
            new_img[rows,cols] = image[int(map_pt[0]),int(map_pt[1])]
            
path = 'C:/Users/user/Desktop/ECE661/HW3/1step'            
cv2.imwrite(os.path.join(path,'1step_Art.jpg'), new_img)
cv2.destroyAllWindows()