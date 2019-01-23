# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import os
import random as rand
import math
import sys

""" Load Image (Resize if Needed) """
def LoadImage():
    path = 'C:/Users/user/Desktop/ECE661/HW5/images/set2'
    img1 = cv2.imread(os.path.join(path,'1.jpg'))
    img2 = cv2.imread(os.path.join(path,'2.jpg'))
    img3 = cv2.imread(os.path.join(path,'3.jpg'))
    img4 = cv2.imread(os.path.join(path,'4.jpg'))
    img5 = cv2.imread(os.path.join(path,'5.jpg'))
    images = [img1, img2, img3, img4, img5]
    for idx in range(0,len(images)):
        if images[idx].shape[0] > 2000 or images[idx].shape[1] > 2000:
            images[idx] = cv2.resize(images[idx],None,fx = 0.25, fy = 0.25)
    return images

""" Direct Linear Transform Homography """
def DLT_Homography(n,srcpt,dstpt):
    eqns = 2*n
    A = np.matrix(np.zeros(shape = (eqns,9)))
    cnt = 0
    for i in range(0, eqns, 2):
        A[i,3:6] = -srcpt[cnt]
        A[i,6:9] = dstpt[cnt][0,1]*srcpt[cnt]
        A[i+1,0:3] = srcpt[cnt]
        A[i+1,6:9] = -dstpt[cnt][0,0]*srcpt[cnt]
        cnt = cnt + 1
    
    u,s,vt = np.linalg.svd(A) 
    h_array = vt[8,0:9]
    H = np.reshape(h_array,(3,3))
    return(H)

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

""" Euclidean Distance Image Correspondence """
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
        match[i,:] = [int(kp1[ref_loc[i]].pt[0]), int(kp1[ref_loc[i]].pt[1]), int(kp2[match_loc[i]].pt[0]),int(kp2[match_loc[i]].pt[1])]
    return match

def Drawkp(image, kp):
    for i in kp:
        pt = (int(i.pt[0]), int(i.pt[1]))
        cv2.circle(image,pt , 5, (0,255,0), 2)
    return image

""" Automatic Homography Generation by RANSAC Algorithm """
def RANSAC_AutoH(p, e, dist_thres, n, corr_pairs):  
    #source point x's and y's
    x1 = corr_pairs[:,0]
    y1 = corr_pairs[:,1]
    #corresponding point x's and y's
    x2 = corr_pairs[:,2]
    y2 = corr_pairs[:,3]
    total_corrs = len(x1)
    
    max_cnt = 0
    N = int(math.log(1-p) / math.log(1-math.pow(1-e,n))) 
    for i in range(0, N):
        srcpt = []
        corrspt = []
        for j in range(0, n):
            rand_idx = rand.randint(0,len(x1)-1)
            srcpt.append(np.matrix([x1[rand_idx],y1[rand_idx],1], dtype = float))
            corrspt.append(np.matrix([x2[rand_idx],y2[rand_idx],1], dtype = float))           
        H = DLT_Homography(n, srcpt, corrspt)
        H = H / H[2,2]
       
        #Detect inliers and exlude outliers
        inliers1 = []
        inliers2 = []
        outliers1 = []
        outliers2 = []
        inlier_cnt = 0
        outlier_cnt = 0
        for k in range(0, total_corrs):
            pt1 = np.matrix([x1[k], y1[k],1], dtype = float)
            pt2 = np.matrix([x2[k], y2[k],1], dtype = float)
            HX = H * pt1.T
            HX /= HX[2]
            dist_err = np.linalg.norm((HX - pt2.T))
            
            if (dist_err < dist_thres):
                match1 = np.matrix([x1[k], y1[k], 1]) 
                match2 = np.matrix([x2[k], y2[k], 1])
                inliers1.append(match1)
                inliers2.append(match2)
                inlier_cnt += 1
            else: 
                match1 = np.matrix([x1[k], y1[k], 1]) 
                match2 = np.matrix([x2[k], y2[k], 1])
                outliers1.append(match1)
                outliers2.append(match2)
                outlier_cnt += 1
                  
        # Update final optimal inliers 
        if (inlier_cnt > max_cnt):
            max_cnt = inlier_cnt
            inliers1_opt = inliers1
            inliers2_opt = inliers2
            outliers1_final = outliers1
            outliers2_final = outliers2
            
    # Refine homography with all inliers 
    print(max_cnt)
    inliers_opt = [inliers1_opt, inliers2_opt]
    outliers_final = [outliers1_final, outliers2_final]
    H_optimal = DLT_Homography(max_cnt, inliers1_opt, inliers2_opt)
    return H_optimal, inliers_opt, outliers_final

""" Construct Match Points for Drawing """
def ConstructMatch(total_pairs, srcpt, corrspt):
    match = np.zeros((total_pairs, 4))    
    for i in range(0, len(srcpt)):
        match[i,:] = [int(srcpt[i][0,0]), int(srcpt[i][0,1]), int(corrspt[i][0,0]),int(corrspt[i][0,1])]
    return match

""" Draw Image Correspondences """
def DrawMatches(img1, img2, match, nonmatch):
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
        cv2.line(img_comb, pt1, pt2, (0,255,0), 1)
        cv2.circle(img_comb, pt1, 5, (0,255,0), 2)
        cv2.circle(img_comb, pt2, 5, (0,255,0), 2)
    if nonmatch is not None:
        for n in nonmatch:
            pt1 = (int(n[0]), int(n[1]))
            pt1 = tuple(pt1)
            pt2 = (int(n[2] + img1.shape[1]),  int(n[3])) 
            pt2 = tuple(pt2)
            cv2.line(img_comb, pt1, pt2, (0,0,255), 1)
            cv2.circle(img_comb, pt1, 5, (0,0,255), 2)
            cv2.circle(img_comb, pt2, 5, (0,0,255), 2)
    return img_comb      
   
def Panoramic_Image(H_list, images):   
    H02 = H_list[0]
    H12 = H_list[1]
    H22 = H_list[2]
    H32 = H_list[3]
    H42 = H_list[4]
    P_02 = np.matmul(H02 , (np.matrix([0,0,1], dtype = float)).T)
    P_02 = P_02/ P_02[2]
    Q_02 = np.matmul(H02 , (np.matrix([images[0].shape[1],0,1], dtype = float)).T)
    Q_02 = Q_02 / Q_02[2]
    R_02 = np.matmul(H02 , (np.matrix([0,images[0].shape[0],1], dtype = float)).T)
    R_02 = R_02 / R_02[2]
    S_02 = np.matmul(H02, (np.matrix([images[0].shape[1],images[0].shape[0],1], dtype = float)).T)
    S_02 = S_02 / S_02[2]
    
    P_42 = np.matmul(H42 , (np.matrix([0,0,1], dtype = float)).T)
    P_42 = P_42 / P_42[2]
    Q_42 = np.matmul(H42 , (np.matrix([images[4].shape[1],0,1], dtype = float)).T)
    Q_42 = Q_42 / Q_42[2]
    R_42 = np.matmul(H42 , (np.matrix([0,images[4].shape[0],1], dtype = float)).T)
    R_42 = R_42 / R_42[2]
    S_42 = np.matmul(H42 , (np.matrix([images[4].shape[1],images[4].shape[0],1], dtype = float)).T)
    S_42 = S_42 / S_42[2]
    
    x_max = np.amax([P_02[0], Q_02[0], R_02[0], S_02[0], P_42[0], Q_42[0], R_42[0], S_42[0]])
    y_max = np.amax([P_02[1], Q_02[1], R_02[1], S_02[1], P_42[1], Q_42[1], R_42[1], S_42[1]])
    x_min = np.amin([P_02[0], Q_02[0], R_02[0], S_02[0], P_42[0], Q_42[0], R_42[0], S_42[0]])
    y_min = np.amin([P_02[1], Q_02[1], R_02[1], S_02[1], P_42[1], Q_42[1], R_42[1], S_42[1]])
    origin = np.array([x_min, y_min])
    x_dim = int(x_max) - int(x_min)
    y_dim = int(y_max) - int(y_min)
    panorama = np.zeros((y_dim, x_dim, 3))
    panorama = Construct_Panorama(np.linalg.inv(H02), panorama, images[0], origin)
    panorama = Construct_Panorama(np.linalg.inv(H12), panorama, images[1], origin)
    panorama = Construct_Panorama(H22, panorama, images[2], origin)
    panorama = Construct_Panorama(np.linalg.inv(H32), panorama, images[3], origin)
    panorama = Construct_Panorama(np.linalg.inv(H42), panorama, images[4], origin)
    return panorama

def Construct_Panorama(Hinv, panorama, srcimg, origin ):
    #scaleX = srcimg.shape[1] / x_dim
    #scaleY = srcimg.shape[0] / y_dim
    for rows in range(0, panorama.shape[0]):
        for cols in range(0, panorama.shape[1]):        
            pt_x = (cols) + origin[0] - 1
            pt_y = (rows) + origin[1] - 1
            new_pt = np.matrix([pt_x,pt_y,1])
            map_pt = np.matmul(Hinv, new_pt.T)
            map_pt = map_pt/map_pt[2]
            if map_pt[0]<srcimg.shape[1]-1 and map_pt[0]>0 and map_pt[1]<srcimg.shape[0]-1 and map_pt[1]>0:
                panorama[rows,cols] = Interpolation(map_pt, srcimg)
    return panorama

""" Bilinear Interpolation for Pixels """
def Interpolation(pt ,srcimg):
    x = pt[0]
    y = pt[1]
    y1 = math.floor(pt[1])
    y2 = math.ceil(pt[1])
    x1 = math.floor(pt[0])
    x2 = math.ceil(pt[0])
    q11 = srcimg[y1,x1]
    q12 = srcimg[y2,x1]
    q21 = srcimg[y1,x2]
    q22 = srcimg[y2,x2]
    f_xy1 = ((x2-x)*q11 / (x2-x1)) + ((x-x1)*q21 / (x2-x1))
    f_xy2 = ((x2-x)*q12 / (x2-x1)) + ((x-x1)*q22 / (x2-x1))
    new_pt = ((y2-y)*f_xy1 / (y2-y1)) + ((y-y1)*f_xy2 / (y2-y1))
    return new_pt

def LM(H, inliers):
    inliers1 = inliers[0]
    inliers2 = inliers[1]
    N = len(inliers[0])
    J = np.zeros((2*N,9))
    Err = np.zeros((2*N))
    for i in range(0,N):
        idx = 2*i
        den = H[2,0]*inliers1[idx][0] + H[2,1]*inliers1[idx][1] + H[2,2]
        num = H[0,0]*inliers1[idx][0] + H[0,1]*inliers1[idx][1] + H[0,2]
        J[idx,0] = inliers1[idx][0] / den
        J[idx,1] = inliers1[idx][1] / den
        J[idx,2] = 1 / den
        J[idx,3] = 0
        J[idx,4] = 0
        J[idx,5] = 0
        J[idx,6] = -(inliers1[idx][0]*num) / (den*den)
        J[idx,7] = -(inliers1[idx][1]*num) / (den*den)
        J[idx,8] = -(num) / (den*den)
        Err[idx] = inliers2[idx][0] - (num/den)
        idx+=1
   
def main():
    images = LoadImage()
    kp , des = SIFT(images)
    #Compute and store adjacent image correspondences
    total_img = len(images)
    euclid_thres = 5
    adj_corrs = []
    for i in range(1, total_img):
        # Forward direction correspondence before center image
        # Reverse direction correspondence after center image
        if i >= math.ceil(total_img/2):
            matches = Euclidean(kp[i], kp[i-1], des[i], des[i-1], euclid_thres)
        else:
            matches = Euclidean(kp[i-1], kp[i], des[i-1], des[i], euclid_thres)
        adj_corrs.append(matches)
        
    sample_size = 6  # Sample size of random selected correspondences
    p = 0.99 # Probability of N trials free of outliers
    e = 0.1 # Initial probability that a correspondence is outlier
    inlier_thres = 30 # Distance threshold for inlier detection
    rand.seed(48)
    H01, inliers01, outliers01 = RANSAC_AutoH(p, e, inlier_thres, sample_size, adj_corrs[0])                              
    H12, inliers12, outliers12 = RANSAC_AutoH(p, e, inlier_thres, sample_size, adj_corrs[1])
    H32, inliers32, outliers32 = RANSAC_AutoH(p, e, inlier_thres, sample_size, adj_corrs[2])
    H43, inliers43, outliers43 = RANSAC_AutoH(p, e, inlier_thres, sample_size, adj_corrs[3])
    
    inliers = ConstructMatch(len(inliers01[0]), inliers01[0], inliers01[1])
    outliers = ConstructMatch(len(outliers01[0]), outliers01[0], outliers01[1])
    ransac01 = DrawMatches(images[0], images[1], inliers, outliers)
    
    inliers = ConstructMatch(len(inliers12[0]), inliers12[0], inliers12[1])
    outliers = ConstructMatch(len(outliers12[0]), outliers12[0], outliers12[1])
    ransac12 = DrawMatches(images[1], images[2], inliers, outliers)
    
    inliers = ConstructMatch(len(inliers32[0]), inliers32[0], inliers32[1])
    outliers = ConstructMatch(len(outliers32[0]), outliers32[0], outliers32[1])
    ransac32 = DrawMatches(images[3], images[2], inliers, outliers)
    
    inliers = ConstructMatch(len(inliers43[0]), inliers43[0], inliers43[1])
    outliers = ConstructMatch(len(outliers43[0]), outliers43[0], outliers43[1])
    ransac43 = DrawMatches(images[4], images[3], inliers, outliers)
    
    path = 'C:/Users/user/Desktop/ECE661/HW5/images/output'
    cv2.imwrite(os.path.join(path,'ransac01.jpg'), ransac01)
    cv2.imwrite(os.path.join(path,'ransac12.jpg'), ransac12)
    cv2.imwrite(os.path.join(path,'ransac32.jpg'), ransac32)
    cv2.imwrite(os.path.join(path,'ransac43.jpg'), ransac43)
    
    # Construct the panorama
    H02 = np.matmul(H01 , H12)
    H22 = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    H42 = np.matmul(H43 , H32)
    H_panorama = [H02, H12, H22, H32, H42]
    panorama = Panoramic_Image(H_panorama, images)
    path = 'C:/Users/user/Desktop/ECE661/HW5/images/output'
    cv2.imwrite(os.path.join(path,'Panorama.jpg'), panorama)
      
main()