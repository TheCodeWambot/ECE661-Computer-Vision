# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:05:23 2018

@author: user
"""
import numpy as np
import cv2
import math
import copy
import os
import scipy.signal as sig
import sys

""" Haar Filter """
def Haar_Filter(sigma): 
    M = math.ceil(4*sigma)
    #M is smallest 'even' integer greater then 4xsigma
    if((M % 2) != 0):
        M = M + 1
    #Create X and Y Haar filters
    kernel = np.ones((M,M), dtype = int)
    kernelX = kernel
    kernelY = copy.deepcopy(kernelX)
    kernelX[:,0:int(M/2)] = -1
    kernelY[0:int(M/2),:] = -1 
    return kernelX, kernelY

""" Extract Local Maxima Corners """
def LocalExtremas(R, window, height, width):           
    #Find interest points as local maxima above threshold 
    threshold_R = R.mean()
    corners = np.zeros((height,width))  
    pad = int((window-1)/2)   
    for rows in range(pad,height-pad):
        for cols in range(pad,width-pad):
            if (R[rows,cols] > 0):
                #Find neighboring ratios
                nbrs = R[rows-pad:rows+pad+1, cols-pad:cols+pad+1]
                maxima = nbrs.max()
                #Assign 1 to pixel coordinate to indicate corner
                if(R[rows,cols] == maxima and R[rows,cols] > threshold_R ):
                    corners[rows,cols] = 1
    return corners
""" Generate List of Dominant Corners """
def Corners(img, knl_X, knl_Y, sigma, window_R):
    #Derive image gradients in X and Y directions
    gradx = sig.convolve2d(img, knl_X, mode = 'same')
    grady = sig.convolve2d(img, knl_Y, mode = 'same')
    Ix = np.divide((gradx - gradx.min()), (gradx.max() - gradx.min()))
    Iy = np.divide((grady - grady.min()), (grady.max() - grady.min()))
    #1st Order Approximations
    Ixx = np.multiply(Ix,Ix)
    Ixy = np.multiply(Ix,Iy)
    Iyy = np.multiply(Iy,Iy)
    #Find summation of pixels within neighborhood
    #Construct matrix C to extract eigenvalues for R
    window = math.ceil(5*sigma)
    if((window % 2) != 0):
        window = window + 1
    pad = int(window/2)
    height = Ixx.shape[0]
    width = Ixx.shape[1]
    C = np.zeros((2,2), dtype = float)
    R = np.zeros((height,width),dtype = float)
    
    for rows in range(pad,height-pad):
        for cols in range(pad,width-pad):
            box_xx = Ixx[rows-pad:rows+pad+1, cols-pad:cols+pad+1] 
            box_xy = Ixy[rows-pad:rows+pad+1, cols-pad:cols+pad+1]
            box_yy = Iyy[rows-pad:rows+pad+1, cols-pad:cols+pad+1]
            sum_xx = np.sum(box_xx)
            sum_xy = np.sum(box_xy)
            sum_yy = np.sum(box_yy)
            C[0,0:2] = [sum_xx, sum_xy]
            C[1,0:2] = [sum_xy, sum_yy]
            det_C = np.linalg.det(C)
            trace_C = np.trace(C)
            #Map a ratio value to each pixel
            #Discard edges lying along edges
            rank = np.linalg.matrix_rank(C)
            if(rank == 2):
                R[rows,cols] = det_C/(trace_C**2)
    cnrlist = LocalExtremas(R, window_R, height, width)
    return cnrlist

""" Normalized Cross Correlation Image Correspondence """
def ImageCorrs(cnrs1, cnrs2, img1, img2, window, method):
    Y1, X1 = cnrs1.nonzero()
    Y2, X2 = cnrs2.nonzero()
    pad = int((window-1)/2)
    cnr1_size = len(X1)
    cnr2_size = len(X2)
    size_list = [cnr1_size, cnr2_size]
    ref_loc = []
    match_loc = []  
    # Choose NCC or SSD for image correspondence
    if (method == 'NCC'):
        corrs = NCC(Y1, X1, Y2, X2, size_list, img1, img2, pad)
        #Maximize NCC for strong correspondence
        for rows in range(0,corrs.shape[0]):
            max_ncc = np.amax(corrs[rows,:])
            if(max_ncc > (0.9*corrs.max())):
                ref_loc.append(rows)
                match_loc.append(np.argmax(corrs[rows,:]))  
    else:
        corrs = SSD(Y1, X1, Y2, X2, size_list, img1, img2, pad) 
        threshold = 10*corrs.min()  
        #Minimize SSD for strong correspondence
        for rows in range(0,corrs.shape[0]):
            min_ssd = np.amin(corrs[rows,:])
            if(min_ssd < threshold):
                ref_loc.append(rows)
                match_loc.append(np.argmin(corrs[rows,:]))               
    match = np.zeros((len(ref_loc),4))   
    for i in range(0,len(ref_loc)):
        match[i,:] = [int(X1[ref_loc[i]]), int(Y1[ref_loc[i]]), int(X2[match_loc[i]]), int(Y2[match_loc[i]])]
    return match

""" Normalized Cross Correlation """
def NCC(Y1, X1, Y2, X2, cnr_size, img1, img2, pad):
    ncc = np.zeros((cnr_size[0],cnr_size[1]), dtype = float)
    for idx1 in range(0,cnr_size[0]):
        for idx2 in range(0,cnr_size[1]):
            #Set up windows of corner neighboring pixels
            nbrs1 = img1[Y1[idx1]-pad:Y1[idx1]+pad+1, X1[idx1]-pad:X1[idx1]+pad+1]
            nbrs2 = img2[Y2[idx2]-pad:Y2[idx2]+pad+1, X2[idx2]-pad:X2[idx2]+pad+1]
            mean1 = float(nbrs1.mean())
            mean2 = float(nbrs2.mean())
            diff1 = nbrs1 - mean1  
            diff2 = nbrs2 - mean2
            norm = float(math.sqrt(np.sum(np.sum(diff1**2))*(np.sum(np.sum(diff2**2)))))          
            #NCC of every corner1 with corner2
            ncc[idx1,idx2] = np.sum(np.sum(np.multiply(diff1,diff2)))/norm
    return ncc

""" Sum of Square Differences """    
def SSD(Y1, X1, Y2, X2, cnr_size, img1, img2, pad):
    ssd = np.zeros((cnr_size[0],cnr_size[1]), dtype = float)
    for idx1 in range(0, cnr_size[0]):
        for idx2 in range(0,cnr_size[1]):
            nbrs1 = img1[Y1[idx1]-pad:Y1[idx1]+pad+1, X1[idx1]-pad:X1[idx1]+pad+1]
            nbrs2 = img2[Y2[idx2]-pad:Y2[idx2]+pad+1, X2[idx2]-pad:X2[idx2]+pad+1]
            diff = np.subtract(nbrs1, nbrs2)
            sq_diff = np.multiply(diff, diff)
            ssd[idx1,idx2] = np.sum(sq_diff)
    return ssd
    
""" Draw Image Correspondences """
def DrawMatches(img1, img2, match):
    #Combine two images horizontally
    new_dim = (max(img1.shape[0],img2.shape[0]), img1.shape[1]+img2.shape[1], 3)
    img_comb = np.zeros(new_dim,type(img1.flat[0]))
    img_comb[0:img1.shape[0],0:img1.shape[1]] = img1
    img_comb[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    for m in match:
        pt1 = (int(m[0]), int(m[1]))
        pt1 = tuple(pt1)
        pt2 = (int(m[2] + img1.shape[1]),  int(m[3])) 
        pt2 = tuple(pt2)
        cv2.line(img_comb, pt1, pt2, (0,0,255), 1)
        cv2.circle(img_comb, pt1, 5, (0,255,0), 2)
        cv2.circle(img_comb, pt2, 5, (0,255,0), 2)
    return img_comb

""" Draw Corners on Merged Image """
def DrawCorners(img1,cnrs1,img2,cnrs2):
    #Combine two images horizontally
    new_dim = (max(img1.shape[0],img2.shape[0]), img1.shape[1]+img2.shape[1], 3)
    img_comb = np.zeros(new_dim)
    img_comb[0:img1.shape[0],0:img1.shape[1]] = img1
    img_comb[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    Y1, X1 = cnrs1.nonzero()
    Y2, X2 = cnrs2.nonzero()
    
    for i in range(0,len(X1)):
        pt1 = (int(X1[i]), int(Y1[i]))
        pt1 = tuple(pt1)
        cv2.circle(img_comb, pt1, 2, (0,255,0), 2)         
    for j in range(0,len(X2)):
        pt2 = (int(X2[j] + img1.shape[1]),  int(Y2[j])) 
        pt2 = tuple(pt2)
        cv2.circle(img_comb, pt2, 2, (0,255,0), 2)
    return img_comb

def main():
    path = 'C:/Users/user/Desktop/ECE661/HW4/pair1'            
    img1 = cv2.imread(os.path.join(path,'1.jpg'))
    #resize to reduce computing time of large images
    if(img1.shape[0] > 2000 or img1.shape[1] > 2000):
        img1 = cv2.resize(img1, None, fx = 0.25, fy = 0.25)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img1_gray = img1_gray.astype(np.float64)
    img2 = cv2.imread(os.path.join(path,'2.jpg'))
    if(img2.shape[0] > 2000 or img2.shape[1] > 2000):
        img2 = cv2.resize(img2, None, fx = 0.25, fy = 0.25)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img2_gray = img2_gray.astype(np.float64)
    
    sigma = 0.5       #scales 0.5, 1.2, 2.5, 3.6
    #define windows centered around pixel (must be odd)
    window_corrs = 21  #neighborhood size for image correspondence
    window_R = 29      #neighborhood size for local maxima of interest points
    method = 'NCC'
    X,Y= Haar_Filter(sigma)
    cnrlist1 = Corners(img1_gray, X, Y, sigma, window_R)
    cnrlist2 = Corners(img2_gray, X, Y, sigma, window_R)
    match = ImageCorrs(cnrlist1,cnrlist2,img1_gray, img2_gray, window_corrs, method)
    #new_img = DrawCorners(img1,cnrlist1,img2,cnrlist2)
    #cv2.imwrite('corners.jpg',new_img)
    new_img = DrawMatches(img1, img2, match)
    path = 'C:/Users/user/Desktop/ECE661/HW4/pair1' 
    cv2.imwrite(os.path.join(path,'wang_0.5SSD.jpg'),new_img)
      
main()
    




