# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 01:28:26 2018

@author: user
"""

import numpy as np
import cv2
import os

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
    path = 'C:/Users/user/Desktop/ECE661/HW4/pair1'            
    img1 = cv2.imread(os.path.join(path,'1.jpg'))
    if(img1.shape[0] > 2000 or img1.shape[1] > 2000):
        img1 = cv2.resize(img1, None, fx = 0.25, fy = 0.25)
    img2 = cv2.imread(os.path.join(path,'2.jpg'))
    if(img2.shape[0] > 2000 or img2.shape[1] > 2000):
        img2 = cv2.resize(img2, None, fx = 0.25, fy = 0.25)

    #Detect Interest Points and Descriptor Vectors with SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #Scale factor for determining threshold
    thresh_scale = 3.5
    match_pts = Euclidean(kp1, kp2, des1, des2, thresh_scale)
    img3 = DrawMatches(img1,img2,match_pts) 
    path = 'C:/Users/user/Desktop/ECE661/HW4/pair1' 
    cv2.imwrite(os.path.join(path,'wang.jpg'),img3)
    
main()

