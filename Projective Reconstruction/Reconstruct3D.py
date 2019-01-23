# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 23:15:24 2018

@author: Hank Huang
"""

import cv2
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as scla
from scipy.optimize import least_squares
import manual_corrs as mc
            
def Normalize_Transform(pts):
    ycoords = pts[:,0]
    xcoords = pts[:,1]
    meanX = np.mean(xcoords)
    meanY = np.mean(ycoords)
    diffX = xcoords-meanX
    diffY = ycoords-meanY
    dist = np.zeros((pts.shape))
    dist[:,0] = diffY
    dist[:,1] = diffX
    mean_dist = np.linalg.norm(dist,axis=1).mean()
    scale = np.sqrt(2) / mean_dist
    x0 = -scale*meanX
    y0 = -scale*meanY
    T = np.matrix([[scale,0,y0],[0,scale,x0],[0,0,1]], dtype = float)
    return T
    
def Estimate_F(pts1_HC, pts2_HC, T1, T2):
    # Normalize pixel correspondences
    X1 = np.dot(T1 , np.transpose(pts1_HC)).T
    X2 = np.dot(T2 , np.transpose(pts2_HC)).T
    tot_corrs = len(pts1_HC)
    A = np.zeros((tot_corrs, 9), dtype=float)
    for i in range(0,tot_corrs):
        x = X1[i][0,0]
        xp = X2[i][0,0]
        y = X1[i][0,1]
        yp = X2[i][0,1]
        A[i,:] = [xp*x, xp*y, xp, yp*x, yp*y, yp, x, y,1]
    U, D, V = np.linalg.svd(A)
    F_initial = V[-1,:]
    F_initial = np.reshape(F_initial, (3,3))
    F_conditioned = Condition_Rank2(F_initial)
    F = T2.T.dot(F_conditioned).dot(T1)  # denormalize F matrix
    F = F / F[2,2]
    return F, X1, X2

""" Condition F to Rank 2 """
def Condition_Rank2(F):
    # Condition F to rank(2)
    U, D_vec, V = np.linalg.svd(F)
    D_vec[-1] = 0
    Dp = np.diag(D_vec)
    F_conditioned =(U.dot(Dp).dot(V))
    return F_conditioned

""" Calculate Epipoles of 2 Images """
def Epipoles(F):
    # Epipoles are right and left null vectors of F
    ep1 = scla.null_space(F)  
    ep2 = scla.null_space(F.T)
    ep2cross = np.matrix([[0,-ep2[2],ep2[1]], [ep2[2],0,-ep2[0]], [-ep2[1],ep2[0],0]], dtype=float) 
    ep1 = ep1 / ep1[2]
    ep2 = ep2 / ep2[2]
    return ep1, ep2, ep2cross
    
""" Triangulate for World Point """
def Triangulate(P1, P2, pts1, pts2):
    tot_corrs = len(pts1)
    Xw_list = []
    A = np.zeros((4,4),dtype = float)
    for i in range(0,tot_corrs):
        x = pts1[i][0]
        y = pts1[i][1]
        xp = pts2[i][0]
        yp = pts2[i][1]
        A[0,:] = np.dot(x,P1[2,:])-P1[0,:]
        A[1,:] = np.dot(y,P1[2,:])-P1[1,:]
        A[2,:] = np.dot(xp,P2[2,:])-P2[0,:]
        A[3,:] = np.dot(yp,P2[2,:])-P2[1,:]
        # Solution is smallest eigenvector of At*A
        #U,D,V = np.linalg.svd(np.dot(A.T,A))
        U,D,V = np.linalg.svd((A))
        Xw = V[-1,:]
        Xw = Xw / Xw[3]
        Xw_list.append(Xw)
    return Xw_list
 
def encode1D(P2):
    params = []
    P2_1d = np.ravel(P2)
    for i in P2_1d:   
        params.append(i)
    return params     

def decode1D(params):
    P2 = np.reshape(params[0:12], (3,4))
    return P2

def cost_func(params, pts1, pts2):
    P2 = decode1D(params)
    P1 = np.hstack((np.identity(3),np.zeros((3,1))))
    Xw = Triangulate(P1,P2,pts1,pts2)

    x1_repj = np.dot(P1, np.transpose(Xw)).T
    x1_repj[:,0] /= x1_repj[:,2]
    x1_repj[:,1] /= x1_repj[:,2]
    x1_repj[:,2] = 1
    x2_repj = np.dot(P2, np.transpose(Xw)).T
    x2_repj[:,0] /= x2_repj[:,2]
    x2_repj[:,1] /= x2_repj[:,2]
    x2_repj[:,2] = 1
    diff1 = pts1 - x1_repj
    diff1 = diff1[:,0:2]
    diff2 = pts2 - x2_repj
    diff2 = diff2[:,0:2]
    error = []
    for rows in range(0,len(diff1)):
            for cols in range(0,2):
                error.append(diff1[rows,cols])
                error.append(diff2[rows,cols])
    #print(max(error))
    return error
 
def Rectification_H(image_dim, pts1, pts2, ep1, ep2, F, P1, P2):
    height = image_dim[0]
    width = image_dim[1]
    
    """ Homography for right camera (2nd image) """
    theta = np.arctan(-(height/2 - ep2[1]) / (width/2 - ep2[0]))
    f = (np.cos(theta)*(-width/2 + ep2[0])) - (np.sin(theta)*(-height/2 + ep2[1]))
    R = np.matrix([[np.cos(theta), -np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]],dtype = float)
    G = np.identity(3)
    G[2,0] = -1/f
    T = np.identity(3)
    T[0,2] = -width/2
    T[1,2] = -height/2
    H2 = np.dot(G,R).dot(T)
    print(np.dot(H2,ep2))
    # Shift the image center to origin by H2
    center_original = np.array([width/2, height/2,1], dtype=float)
    center_proj = np.dot(H2,center_original.T)
    center_rect2 = center_proj / center_proj[0,2]
    
    # Homography to shift origin back to initial image center
    T2_reverse = np.matrix([[1,0,(width/2)-center_rect2[0,0]], [0,1,(height/2)-center_rect2[0,1]],[0,0,1]], dtype=float)
    H2_reverse = np.dot(T2_reverse,H2)
    
    """ Homography for left camera (1st image) """

    theta = np.arctan(-(height/2 - ep1[1]) / (width/2 - ep1[0]))
    f = (np.cos(theta)*(-width/2 + ep1[0])) - (np.sin(theta)*(-height/2 + ep1[1]))
    R = np.matrix([[np.cos(theta), -np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]],dtype = float)
    G = np.identity(3)
    G[2,0] = -1/f
    T = np.identity(3)
    T[0,2] = -width/2
    T[1,2] = -height/2
    H1 = np.dot(G,R).dot(T)
    print(np.dot(H1,ep1))
    # Shift the image center to origin by H1
    center_rect1 = np.dot(H1,center_original)
    center_rect1 = center_rect1 / center_rect1[0,2]
    # Homography to shift origin back to initial image center 
    T1_reverse = np.matrix([[1,0,(width/2)-center_rect1[0,0]], [0,1,(height/2)-center_rect1[0,1]],[0,0,1]], dtype=float)
    H1_reverse = T1_reverse.dot(H1)
    
    """ Compute Rectified Epipolar Parameters """
    H1_inv = np.linalg.inv(H1_reverse)
    H2_inv = np.linalg.inv(H2_reverse)
    F_rect = H2_inv.T.dot(F).dot(H1_inv)
    pts1_rect = H1_reverse.dot(np.transpose(pts1)).T
    pts1_rect[:,0] /= pts1_rect[:,2]
    pts1_rect[:,1] /= pts1_rect[:,2]
    pts1_rect[:,2] = 1
    pts2_rect = H2_reverse.dot(np.transpose(pts2)).T
    pts2_rect[:,0] /= pts2_rect[:,2]
    pts2_rect[:,1] /= pts2_rect[:,2]
    pts2_rect[:,2] = 1
    ep1_rect, ep2_rect, ep2cross = Epipoles(F_rect)
    return H1_reverse, H2_reverse, pts1_rect, pts2_rect, ep1_rect, ep2_rect, F_rect

""" Generate Rectified Images """
def Rectify_Image(image, H):
    height = image.shape[0]
    width = image.shape[1]
    img_cnrs = []
    img_cnrs.append(np.array([0,0,1]))
    img_cnrs.append(np.array([width,0,1]))
    img_cnrs.append(np.array([0,height,1]))
    img_cnrs.append(np.array([width,height,1]))
    cnrs_rect1, dim, origin = PointProjection(H, img_cnrs)
    height_rect = dim[1] - origin[1]
    width_rect = dim[0] - origin[0]
    Scale = np.matrix([[width/width_rect,0,0],[0,height/height_rect,0],[0,0,1]])
    H_scaled = Scale.dot(H)
    cnrs_rect2, dim, origin = PointProjection(H_scaled, img_cnrs)
    
    T = np.matrix([[1,0,-origin[0]+1],[0,1,-origin[1]+1],[0,0,1]])
    H_final = T.dot(H_scaled)
    H_final_inv = np.linalg.inv(H_final)
    
    height_rect = int(dim[1] - origin[1])
    width_rect = int(dim[0] - origin[0])
    scale_y = float(height) / float(width_rect)
    scale_x = float(width) / float(height_rect)
    if scale_y > scale_x:
        scale = scale_y
    else:
        scale = scale_x
    h_new = int(height*scale)
    w_new = int(width*scale)
    image_rect = np.zeros(((h_new),(w_new),3))
    #image_rect = np.zeros((height_rect,width_rect,3))
    print(image_rect.shape)
    for rows in range(0, image_rect.shape[0]):
        for cols in range(0, image_rect.shape[1]):        
            pt_x = cols/scale -origin[0] -1
            pt_y = rows/scale -origin[1] -1
            new_pt = np.matrix([pt_y,pt_x,1])
            map_pt = np.matmul(H_final_inv, new_pt.T)
            map_pt = map_pt/map_pt[2]
            if map_pt[0]<image.shape[0]-1 and map_pt[0]>0 and map_pt[1]<image.shape[1]-1 and map_pt[1]>0:
                image_rect[rows,cols] = Interpolation(map_pt, image)
    
    return image_rect

""" Apply Homography on Corner Pixels """
def PointProjection(H, pts):
    #Apply homography on points
    p_w = np.dot(H , pts[0].T)
    p_w = p_w / p_w[0,2]
    p_w[0,0] = int(p_w[0,0])
    p_w[0,1] = int(p_w[0,1])
    q_w = np.dot(H , pts[1].T)
    q_w = q_w / q_w[0,2]
    q_w[0,0] = int(q_w[0,0])
    q_w[0,1] = int(q_w[0,1])
    r_w = np.dot(H , pts[2].T)
    r_w = r_w / r_w[0,2]
    r_w[0,0] = int(r_w[0,0])
    r_w[0,1] = int(r_w[0,1])
    s_w = np.dot(H , pts[3].T)
    s_w = s_w / s_w[0,2]
    s_w[0,0] = int(s_w[0,0])
    s_w[0,1] = int(s_w[0,1]) 
    pts_new = np.array([p_w,q_w,r_w,s_w])
    
    dim_x = np.amax([p_w[0,1], q_w[0,1], r_w[0,1], s_w[0,1]])
    dim_y = np.amax([p_w[0,0], q_w[0,0], r_w[0,0], s_w[0,0]])
    dimensions = np.array([dim_x,dim_y])
    origin_x = np.amin([p_w[0,1], q_w[0,1], r_w[0,1], s_w[0,1]])
    origin_y = np.amin([p_w[0,0], q_w[0,0], r_w[0,0], s_w[0,0]])
    origin = np.array([origin_x,origin_y])
    return pts_new, dimensions, origin

""" Bilinear Interpolation for Pixels """
def Interpolation(pt ,srcimg):
    x = pt[1]
    y = pt[0]
    y1 = math.floor(pt[0])
    y2 = math.ceil(pt[0])
    x1 = math.floor(pt[1])
    x2 = math.ceil(pt[1])
    q11 = srcimg[y1,x1]
    q12 = srcimg[y2,x1]
    q21 = srcimg[y1,x2]
    q22 = srcimg[y2,x2]
    f_xy1 = ((x2-x)*q11 / (x2-x1)) + ((x-x1)*q21 / (x2-x1))
    f_xy2 = ((x2-x)*q12 / (x2-x1)) + ((x-x1)*q22 / (x2-x1))
    new_pt = ((y2-y)*f_xy1 / (y2-y1)) + ((y-y1)*f_xy2 / (y2-y1))
    return new_pt

""" SIFT Keypoint Detection """
def SIFT(image_list):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = []
    descriptors = []
    for idx in range(0, len(image_list)):
        kp, des = sift.detectAndCompute(image_list[idx],None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors

""" Euclidean Distance of Descriptor Vectors """
def Euclidean(kp1, kp2, des1, des2, thresh_scale):
    distance = np.zeros((len(des1), len(des2)))
    for idx1 in range(0, len(des1)):
        for idx2 in range (0, len(des2)):
            distance[idx1,idx2] = np.sqrt(np.sum((des1[idx1] - des2[idx2])**2))   
    ref_loc = []
    match_loc = []
    #Minimize distance for strongest correspondence
    for i in range(0,distance.shape[0]):     
        min_dist = np.amin(distance[i,:])
        if min_dist < thresh_scale*distance.min():
            ref_loc.append(i)
            match_loc.append(np.argmin(distance[i,:]))
             
    match = np.zeros((len(ref_loc), 4))   
    for i in range(0,len(ref_loc)):
        match[i,:] = [kp1[ref_loc[i]].pt[0], kp1[ref_loc[i]].pt[1], kp2[match_loc[i]].pt[0],kp2[match_loc[i]].pt[1]]
    return match

""" Draw Image Correspondences """
def DrawMatches(img1, img2, match):
    #Combine two images horizontally
    new_dim = (max(img1.shape[0],img2.shape[0]), img1.shape[1]+img2.shape[1], 3)
    img_comb = np.zeros(new_dim,type(img1.flat[0]))
    img_comb[0:img1.shape[0],0:img1.shape[1]] = img1
    img_comb[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2 
    #Circle interest points and connect corresponding pairs
    for m in match:
        pt1 = (int(m[0]), int(m[1]))
        pt1 = tuple(pt1)
        pt2 = (int(m[2] + img1.shape[1]),  int(m[3])) 
        pt2 = tuple(pt2)
        cv2.line(img_comb, pt1, pt2, (0,0,255), 1)
        cv2.circle(img_comb, pt1, 5, (0,255,0), 2)
        cv2.circle(img_comb, pt2, 5, (0,255,0), 2)
    return img_comb   
  
def main():
    # Get Manual Correspondences
    image1, image2, img1_pts , img2_pts = mc.Manual_Correspondence()
    #Convert to HC representation
    pts1_HC = [[pt[0], pt[1], 1] for pt in img1_pts]
    pts2_HC = [[pt[0], pt[1], 1] for pt in img2_pts]
   
    T1 = Normalize_Transform(img1_pts)
    T2 = Normalize_Transform(img2_pts)
    
    F, pts1_norm, pts2_norm = Estimate_F(pts1_HC, pts2_HC, T1, T2)
    print("Rank of Initial F: %d" % np.linalg.matrix_rank(F))
    ep1, ep2, ep2cross = Epipoles(F)
    print(F)
    print(ep1)
    print(ep2)
    # Camera Canonical Configuration
    P1 = np.hstack((np.identity(3),np.zeros((3,1))))
    P2 = np.hstack((ep2cross.dot(F), ep2))
    print(P1)
    print(P2)
    # Nonlinear Least Square Optimization of F
    params = encode1D(P2)
    #cost_func(params,img1_pts,img2_pts)
    params_opt = least_squares(cost_func, params, args=(pts1_HC,pts2_HC),method='lm')
    P2_opt = decode1D(params_opt.x)
    #P2_opt = decode1D(params_opt[0])
    C = np.array([0,0,0,1])
    ep_temp = P2_opt.dot(C)
    ep2cross = np.matrix([[0,-ep_temp[2],ep_temp[1]], [ep_temp[2],0,-ep_temp[0]], [-ep_temp[1],ep_temp[0],0]], dtype=float)
    F_temp = ep2cross.dot(P2_opt).dot(np.linalg.pinv(P1))   # F = [ex]P'*P+
    F_opt = Condition_Rank2(F_temp)
    F_opt = np.dot(np.dot(T2.T, F_opt), T1)
    
    F_opt = F_opt / F_opt[2,2]
    print("Rank of Optimized F: %d" % np.linalg.matrix_rank(F_opt))
    ep1_opt, ep2_opt, ep2cross_opt = Epipoles(F_opt)
    print(F_opt)
    print(ep1_opt)
    print(ep2_opt)
    print(P1)
    print(P2_opt)
    image_dim = [image1.shape[0],image1.shape[1]]
    H1, H2, pts1_rect, pts2_rect, ep1_rect, ep2_rect, F_rect = Rectification_H(image_dim,pts1_HC,pts2_HC,ep1_opt,ep2_opt,F_opt,P1,P2_opt)
    print(F_rect)
    print(ep1_rect)
    print(ep2_rect)
    if not os.path.isfile("img1_rect.jpg") and os.path.isfile("img2_rect.jpg"):
        image1_rect = Rectify_Image(image1, H1)
        cv2.imwrite("img1_rect.jpg",image1_rect)  
        image2_rect = Rectify_Image(image2, H2)
        cv2.imwrite("img2_rect.jpg",image2_rect)
    else:
        image1_rect = cv2.imread("img1_rect.jpg")
        image2_rect = cv2.imread("img2_rect.jpg")
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1_rect,None)
    kp2, des2 = sift.detectAndCompute(image2_rect,None)
    thresh_scale = 3
    match_pts = Euclidean(kp1, kp2, des1, des2, thresh_scale)
    image_sift = DrawMatches(image1_rect,image2_rect,match_pts) 
    outpath = os.path.join(os.getcwd(), "output")
    cv2.imwrite(os.path.join(outpath,'img_rect_sift.jpg'),image_sift)
    
    sift_left = match_pts[:,0:2]
    sift_right = match_pts[:,2:4]
    
    pts1_new = [[pt[0], pt[1], 1] for pt in sift_left]
    pts2_new = [[pt[0], pt[1], 1] for pt in sift_right]
    
    T1 = Normalize_Transform(pts1_new)
    T2 = Normalize_Transform(pts2_new)
    
    F, pts1_norm, pts2_norm = Estimate_F(pts1_new, pts2_new, T1, T2)
    ep1, ep2, ep2cross = Epipoles(F)
    # Camera Canonical Configuration
    P1 = np.hstack((np.identity(3),np.zeros((3,1))))
    P2 = np.hstack((ep2cross.dot(F), ep2))
    # Nonlinear Least Square Optimization of F
    params = encode1D(P2)

    params_opt = least_squares(cost_func, params, args=(pts1_HC,pts2_HC),method='lm')
    P2_opt = decode1D(params_opt.x)
    World = Triangulate(P1, P2_opt, pts1_new, pts2_new)
    
    plt.figure()
    for point in World:
        Axes3D.scatter(World[1],World[0],World[2], 'r')
main()