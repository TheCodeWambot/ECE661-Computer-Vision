# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:17:51 2018

@author: Hank Huang
"""

import cv2
import numpy as np
import os
import sys
import copy
from scipy.optimize import leastsq

def LineDetect(image,outpath,filename):
    height = image.shape[0]
    width = image.shape[1]
    dim = [height, width]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 255*1.5, 255)
    #cv2.imwrite(os.path.join(outpath,filename),edges)
    """cv2.imshow('edges',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
    return LineFilter(lines, dim)

def printLines(image, lines,outpath, filename):
    for idx in range(0, len(lines)):
        for rho,theta in lines[idx]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),1) 
    cv2.imwrite(os.path.join(outpath,filename),image)
    """cv2.imshow('lines',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """
         
""" Filter Out Additional Lines from Hough Transform """     
def LineFilter(lines, dim):
    vert = []
    horiz = []
    # Separate horizontal and vertical lines by slope
    for l in lines:
        pt1, pt2 = Polar2Cartesian(l, 1000)
        if (abs(pt2[1] - pt1[1])) > 1500:
        #if (abs(l[0,1]) < np.pi/4):
            vert.append(l)
        else: 
            horiz.append(l)
    # Filter out extra vertical and horizontal lines by intersections 
    badvert = []
    for i in range(0, len(vert)):   
        for j in range(0, len(vert)):
            if i != j:
                intersect = Intersect(vert[i], vert[j])    
                if (intersect[0]>0 and intersect[0]<dim[1] and intersect[1]>0 and intersect[1]<dim[0]):
                    if j not in badvert and j > i: 
                        badvert.append(j)  
                    else:
                        badvert.append(i)  # Case: close parallel lines
                                                                      
    new_vert = []
    for idx in range(0,len(vert)):
        if idx not in badvert:
            new_vert.append(vert[idx])
            
    badhoriz = []
    for i in range(0, len(horiz)):
        for j in range(0, len(horiz)):
            if i != j:
                intersect = Intersect(horiz[i], horiz[j])
                if (intersect[0]>0 and intersect[0]<dim[1] and intersect[1]>0 and intersect[1]<dim[0]):
                    if j not in badhoriz and j > i:
                        badhoriz.append(j)
                    else:
                        badhoriz.append(i)
    new_horiz = []
    for idx in range(0,len(horiz)):
        if idx not in badhoriz:
            new_horiz.append(horiz[idx])
            
    new_lines = new_vert + new_horiz
    new_vert = sorted(new_vert, key = lambda polar: polar[0,0]*np.cos(polar[0,1]))   
    new_horiz = sorted(new_horiz, key = lambda polar: polar[0,0]*np.sin(polar[0,1]))
    return new_lines, new_vert, new_horiz

def Polar2Cartesian(pcoord, dist):
    rho = pcoord[0,0]
    theta = pcoord[0,1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + dist*(-b))
    y1 = int(y0 + dist*(a))
    x2 = int(x0 - dist*(-b))
    y2 = int(y0 - dist*(a))
    pt1 = np.array([x1, y1, 1], dtype=float)
    pt2 = np.array([x2, y2, 1], dtype=float)
    return pt1, pt2

def Intersect(line1, line2):
    pt1, pt2 = Polar2Cartesian(line1, 1000)
    pt3, pt4 = Polar2Cartesian(line2, 1000)
    l1 = np.cross(pt1, pt2)
    l2 = np.cross(pt3, pt4)
    intersect = np.cross(l1, l2)
    norm = intersect / intersect[2]
    return norm
    
def CornerDetect(vert, horiz, image):
    corner = []
    for l1 in horiz:
        for l2 in vert:
            corner.append(Intersect(l1, l2))
            intersect = Intersect(l1, l2)
            cv2.circle(image, (int(intersect[0]),int(intersect[1])),3, (0,0,255),-1)
    """cv2.imshow('corners',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  """      
    return corner      

def LabelCorner(cnrs, image,outpath,filename):
    total = len(cnrs)
    for i in range(0, total):
        text = str(i+1)
        cv2.putText(image, text, (int(cnrs[i][0]),int(cnrs[i][1])), cv2.FONT_HERSHEY_COMPLEX,0.3,(255,0,0))
    cv2.imwrite(os.path.join(outpath,filename),image)
    """cv2.imshow('labels',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  """ 
  
def WorldCoord(dim, length):  
    WC = []
    x = 0
    y = 0
    for row in range(0,dim[0]):
        x = 0 
        for col in range(0,dim[1]):
            pt = np.array([x,y,1], dtype = float)
            WC.append(pt)
            x += length
        y += length  
    return WC


""" Direct Linear Transform Homography """
def DLT_Homography(n,srcpt,dstpt):
    eqns = 2*n
    A = np.matrix(np.zeros(shape = (eqns,9)))
    cnt = 0
    for i in range(0, eqns, 2):
        A[i,3:6] = srcpt[cnt]
        A[i,6:9] = -dstpt[cnt][1]*srcpt[cnt]
        A[i+1,0:3] = srcpt[cnt]
        A[i+1,6:9] = -dstpt[cnt][0]*srcpt[cnt]
        cnt = cnt + 1
    u,s,v = np.linalg.svd(A) 
    minidx = s.argmin()
    h_array = v[minidx]
    H = np.reshape(h_array,(3,3))
    return(H)    
  
""" Calculate Absolute Conic Image """
def AbsConicImg(H, V_list):
    h1 = H[:,0]        
    h2 = H[:,1]
    v12 = np.array([h1[0]*h2[0], (h1[0]*h2[1]+h1[1]*h2[0]), h1[1]*h2[1], (h1[2]*h2[0]+h1[0]*h2[2]), (h1[2]*h2[1]+h1[1]*h2[2]), h1[2]*h2[2]])
    v11 = np.array([h1[0]*h1[0], (h1[0]*h1[1]+h1[1]*h1[0]), h1[1]*h1[1], (h1[2]*h1[0]+h1[0]*h1[2]), (h1[2]*h1[1]+h1[1]*h1[2]), h1[2]*h1[2]])
    v22 = np.array([h2[0]*h2[0], (h2[0]*h2[1]+h2[1]*h2[0]), h2[1]*h2[1], (h2[2]*h2[0]+h2[0]*h2[2]), (h2[2]*h2[1]+h2[1]*h2[2]), h2[2]*h2[2]])
    v12 = np.squeeze(v12, axis = (2,))
    v11 = np.squeeze(v11, axis = (2,))
    v22 = np.squeeze(v22, axis = (2,))
    V = np.vstack((v12.T, np.subtract(v11,v22).T))
    # Accumulate n camera views to solve W
    if len(V_list) == 0:
        V_list.append(V)
        V_list = np.squeeze(V_list)
    else:       
        V_list = np.squeeze(V_list)
        V_list = np.vstack((V_list, V))
    u,d,v = np.linalg.svd(V_list)
    W = v[5,0:6]
    return W.T , V_list  

def Intrinsic(W):
    y0 = (W[1]*W[3] -W[0]*W[4]) / (W[0]*W[2] - W[1]**2)
    lmbda = W[5] - (((W[3]**2)+ y0*(W[1]*W[3]-W[0]*W[4])) / W[0])
    alpha_x = np.sqrt(lmbda / W[0])
    alpha_y = np.sqrt((lmbda*W[0]) / (W[0]*W[2] - W[1]**2))
    s = -(W[1]*(alpha_x**2)*alpha_y) / lmbda
    x0 = (s*y0 / alpha_y) - ((W[3]*alpha_x**2)/ lmbda)
    K = np.zeros((3,3), dtype = float)
    K[0,0] = alpha_x
    K[0,1] = s
    K[0,2] = x0
    K[1,1] = alpha_y
    K[1,2] = y0
    K[2,2] = 1
    return K
  
def Extrinsic(H, K):  
    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]
    K_inv = np.linalg.inv(K)
    mag_scale = 1/np.linalg.norm(K_inv*h1) 
    t = np.dot(K_inv,h3)
    if t[2] < 0:
        mag_scale = -mag_scale
    r1 = np.dot(K_inv,h1)*mag_scale
    r2 = np.dot(K_inv,h2)*mag_scale
    r3 = np.cross(r1.T,r2.T)
    t = t*mag_scale
    R = np.zeros((3,3), dtype = float)            
    R[:,0] = r1.T
    R[:,1] = r2.T
    R[:,2] = r3
    # Orthonormal approximation for R
    u, s, vt = np.linalg.svd(R)
    R_ortho = np.dot(u,vt)
    return R_ortho, t
 
""" Transform Rotation Parameters for Optimization """
def Rodrigues(rot, reverse):
    if reverse == False:
        phi = np.arccos(((np.matrix.trace(rot)-1) / 2))
        scale = (phi / (2*np.sin(phi))) 
        w_vector = np.array([rot[2,1]-rot[1,2], rot[0,2]-rot[2,0], rot[1,0]-rot[0,1]], dtype = float)
        w_vector = (w_vector * scale).T   
        return w_vector
    else:
        I = np.identity(3)
        wx = np.zeros((3,3),dtype =float)
        wx[0,1] = -rot[2]
        wx[0,2] = rot[1]
        wx[1,0] = rot[2]
        wx[1,2] = -rot[0]
        wx[2,0] = -rot[1]
        wx[2,1] = rot[0]
        phi = np.linalg.norm(wx)
        R_matrix = I + ((np.sin(phi)/phi)*wx) + (((1-np.cos(phi))/phi**2)*(wx**2))      
        return R_matrix
    
""" Encode Intrinsic/Extrinsic Parameters to 1D Array """
def Encode1D(K, w_list, t_list):
    p = np.array([K[0,0],K[0,1],K[0,2],K[1,1],K[1,2]], dtype = float)
    for i in range(len(w_list)):    
        p = np.append(p,w_list[i][0])
        p = np.append(p,w_list[i][1])
        p = np.append(p,w_list[i][2])
        p = np.append(p,t_list[i][0])
        p = np.append(p,t_list[i][1])
        p = np.append(p,t_list[i][2])      
    return p

""" Decode Parameters from 1D Representation """
def Decode1D(p):
    K = np.zeros((3,3), dtype = float)
    K[0,0] = p[0]
    K[0,1] = p[1]
    K[0,2] = p[2]
    K[1,1] = p[3]
    K[1,2] = p[4]
    K[2,2] = 1
    R_list = []
    t_list = []
    for i in range(5,len(p),6):
        if i == len(p):
            break
        w = p[i:i+3]
        R = Rodrigues(w, True)
        R_list.append(R)
        t = np.array([p[i+3:i+6]],dtype=float)
        t_list.append(t.T)  
    return K, R_list, t_list

def DGeom(p, cnrsW, cnrsImg, totimg):
    K = np.zeros((3,3), dtype = float)
    K[0,0] = p[0]
    K[0,1] = p[1]
    K[0,2] = p[2]
    K[1,1] = p[3]
    K[1,2] = p[4]
    K[2,2] = 1
    cnt = 5 
    xW = [np.array([cnr[0], cnr[1], 0, 1], dtype=float) for cnr in cnrsW] 
    error = []
    for i in range(0, totimg):
        w = p[cnt:cnt+3]
        R = Rodrigues(w,True)
        t = np.array([p[cnt+3:cnt+6]])  
        cnt = cnt + 6
        Rt = np.hstack((R,t.T))
        P = np.dot(K,Rt)
        x_new = (np.dot(P,np.transpose(xW))).T
        x_new[:,0] /= x_new[:,2]
        x_new[:,1] /= x_new[:,2]
        x_new = x_new[:,0:2]
        x_img = np.squeeze(cnrsImg[i])[:,0:2]   
        
        diff = x_img - x_new
        for rows in range(0,len(diff)):
            for cols in range(0,2):
                error.append(diff[rows][cols])
            
    print(max(error))
    return error

def Proj_Homography(P):
    R = P[:,0:2]
    t = P[:,3]
    t = np.array([P[:,3]],dtype=float)
    H_proj = np.hstack((R,t.T))
    return H_proj

def Reproject(fixed,imgnames,images,P,cnrs_img, outpath):
    fidx = imgnames.index(fixed)
    for idx in range(0,len(imgnames)):
        xImg = [np.array([cnr[0], cnr[1], 1], dtype=float) for cnr in cnrs_img[idx]] 
        fimage = images[fidx].copy()
        Pview = Proj_Homography(P[idx])
        Pfix = Proj_Homography(P[fidx])
        P_inv = np.linalg.pinv(Pview)
        x_proj = (np.dot(P_inv,np.transpose(xImg)))
        x_proj = np.dot(Pfix,x_proj).T
        x_proj[:,0] /= x_proj[:,2]
        x_proj[:,1] /= x_proj[:,2]
        x_proj = x_proj[:,0:2]
        for pt in x_proj:        
            cv2.circle(fimage, (int(pt[0]),int(pt[1])),3, (0,255,0),-1)
        cv2.imwrite(os.path.join(outpath,imgnames[idx]),fimage)
        """cv2.imshow('reproject',fimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """   
    return

def main():
    curdir = os.getcwd()
    outdir = 'Output'
    outpath = os.path.join(curdir, outdir)
    if not os.path.exists(outpath):
        os.makedirs(outdir)
    datadir = os.path.join(curdir,'Dataset2')
    imglist = os.listdir(datadir)
    
    H_list = []
    V_list = []
    cnrs_list = []
    imgbuff = []
    """ Absolute Conic Image """
    for filename in imglist:
        image = cv2.imread(os.path.join(datadir, filename))
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            image = cv2.resize(image, None, fx = 0.20, fy = 0.20)
        lines, vert, horiz = LineDetect(image,outpath,filename)
        #printLines(image, lines,outpath,filename)
        cnrs_img = CornerDetect(vert,horiz, image)
        cnrs_list.append(cnrs_img)
        LabelCorner(cnrs_img, image,outpath,filename)
        dim = [len(horiz), len(vert)]
        cnrs_world = WorldCoord(dim, 25)  # 25mm between each corner
        H = DLT_Homography(len(cnrs_img), cnrs_world, cnrs_img)
        H_list.append(H)
        W, V_list = AbsConicImg(H, V_list)    
        imgbuff.append(image)
        
    """ Intrinsic Parameters K """
    K = Intrinsic(W)  
    print(K)
    
    """ Extrinsic Parameters R and t """
    R_preLM = []
    t_preLM = []
    P_preLM = []
    w_list = []
    for idx in range(0, len(imglist)):
        R, t = Extrinsic(H_list[idx], K)
        R_preLM.append(R)
        t_preLM.append(t.T)
        # Projection Matrix before Nonlinear Optimizationd
        Rt = np.hstack((R,t))
        P_preLM.append(np.dot(K,Rt))
        # Convert R to Rodrigues Representation for LM
        w = Rodrigues(R, False)      
        w_list.append(w)
    Reproject('Pic_13.jpg',imglist,imgbuff,P_preLM,cnrs_list,outpath)  
    
    """ Nonlinear Optimization """
    t_list = np.squeeze(t_preLM)
    p = Encode1D(K, w_list, t_list) 
    # If pre-optimized, then just load parameters from file
    opt_flag = 0
    if opt_flag == 0:
        print("Start Optimize")
        p_opt = leastsq(DGeom, p, args=(cnrs_world,cnrs_list,len(imglist)))
        np.savetxt("camparams.txt",p_opt[0])
        K_opt,R_opt,t_opt = Decode1D(p_opt[0])
    else:
        p_opt = np.loadtxt("camparams.txt",dtype=float)
        K_opt,R_opt,t_opt = Decode1D(p_opt)
    print(K_opt)
    P_postLM = []
    for i in range(0, len(imglist)):
        Rt_opt = np.hstack((R_opt[i],t_opt[i]))
        P_new = np.dot(K_opt,Rt_opt)
        P_postLM.append(P_new)
    Reproject('Pic_13.jpg',imglist,imgbuff,P_postLM,cnrs_list,outpath)
        
main()
