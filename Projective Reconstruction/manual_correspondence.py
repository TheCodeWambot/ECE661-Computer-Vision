# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:31:25 2018

@author: Hank Huang
"""

import cv2
import numpy as np
import os

""" Select Points from Camera Views """
def Select_Pts():
    cv2.namedWindow("Correspondence", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Correspondence",img_concat.shape[1],img_concat.shape[0])
    cv2.imshow("Correspondence", img_concat)
    cv2.setMouseCallback("Correspondence", mouse_event)
    cv2.waitKey(0)
    while len(img1_select) != len(img2_select):
        print("Error: Uneven Correspondences")
        Select_Pts()
    cv2.destroyAllWindows()
    print("Total Corresponding Pairs: %d" % len(img1_select))
    return

def mouse_event(event,x,y,flags,param):
    global img1_select, img2_select
    img_width = int(img_concat.shape[1] / 2)
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < img_width:
            img1_select.append([x,y])
            cv2.circle(img_concat,(x,y),4,(255,0,0),-1)
            cv2.imshow("Correspondence", img_concat)
        else:
            img2_select.append([x-img_width,y])
            cv2.circle(img_concat,(x,y),4,(0,0,255),-1)
            cv2.imshow("Correspondence", img_concat)
    
def Manual_Correspondence():        
    curdir = os.getcwd()
    imgdir = os.path.join(curdir, "images")
    image1 = cv2.imread(os.path.join(imgdir,"cube1.jpg"))
    image1 = cv2.resize(image1, None, fx = 0.25, fy = 0.25)
    image2 = cv2.imread(os.path.join(imgdir,"cube2.jpg"))
    image2 = cv2.resize(image2, None, fx = 0.25, fy = 0.25)
    
    global img_concat
    img_concat = np.hstack((image1,image2))
    
    global img1_select, img2_select
    img1_select = []
    img2_select = []
    
    if os.path.isfile("manual_corrs.txt"):
        points = np.loadtxt("manual_corrs.txt", dtype = int)
        img1_pts = np.array(points[0:int(len(points)/2)])
        img2_pts = np.array(points[int(len(points)/2):])   
    else:
        Select_Pts()  # Manually Click Correspondences
        selected_pts = np.vstack((img1_select,img2_select))
        np.savetxt("manual_corrs_cube.txt",selected_pts)
        img1_pts = np.array(img1_select)
        img2_pts = np.array(img2_select) 
        cv2.imwrite("img_concat_cube.jpg", img_concat)
        
    return image1, image2, img1_pts, img2_pts      
            