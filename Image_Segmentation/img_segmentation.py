# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 22:48:48 2018

@author: hank huang
"""
import numpy as np
import cv2
import os
import sys
import math

def Otsu(img, refines):   
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grayimg = grayimg.astype(np.float64)
    L = 256
    # gray level probability density 
    glvl_pdf = []
    glvl_pixels = []
    # Total number of pixels
    N = grayimg.shape[0] * grayimg.shape[1]
    for lvl in range(0,L):
        glvl_size = len((np.where(grayimg == lvl))[0])
        glvl_pixels.append(glvl_size)
        glvl_pdf.append(float(glvl_size / N))
    
    k = 0  # initial segmentation threshold
    overall_k = 0
    max_sig2b = 0 # initial max between class variance
    
    for N_iter in range(0,refines+1):
        for temp_k in range(k,L):
            # w0: background class prob , w1: object class prob
            w0 = float(np.sum(glvl_pdf[k:temp_k+1]))
            w1 = float(np.sum(glvl_pdf[temp_k+1:L]))
            if w0 > 0 and w1 > 0:
                mu0 = np.sum(np.multiply(np.arange(k,temp_k+1) , glvl_pdf[k:temp_k+1])) / w0
                mu1 = np.sum(np.multiply(np.arange(temp_k+1,L) , glvl_pdf[temp_k+1:L])) / w1
                sig2b = w0*w1*(mu1-mu0)**2
                if(sig2b > max_sig2b):
                    k = temp_k
                    max_sig2b = sig2b
                    
        # Refine threshold with foreground probability density function
        fgnd_size = np.sum(glvl_pixels[k+1:L])
        for lvl in range(0,k+1):
            glvl_pdf[lvl] = 0
        for lvl in range(k+1,L):
            glvl_size = len((np.where(grayimg == lvl))[0])
            glvl_pdf[lvl] = float(glvl_size / fgnd_size)
        
        overall_k = k    
    seg_mask = np.zeros((grayimg.shape[0],grayimg.shape[1]), dtype = np.uint8)  
    for cols in range(0,seg_mask.shape[0]):
        for rows in range(0, seg_mask.shape[1]):
            if (grayimg[cols,rows] > overall_k):
                seg_mask[cols,rows] = 255                
    return seg_mask

def RGB_Segmentation(img, refines):  
    # Create individual images of each color channel
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]
    BGR = [0,1,2]
    img_buffB = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    img_buffG = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    img_buffR = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    img_buffB[:,:,BGR[0]] = blue
    img_buffG[:,:,BGR[1]] = green
    img_buffR[:,:,BGR[2]] = red
    blue_mask = Otsu(img_buffB, refines[0])
    green_mask = Otsu(img_buffG, refines[1])
    red_mask = Otsu(img_buffR, refines[2])                                              
    return blue_mask, green_mask, red_mask

def Texture_Segmentation(img, windows, refines):
    # Create individual images of each texture N
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    textmap3 = Texture_Map(gray, windows[0])
    textmap5 = Texture_Map(gray, windows[1])
    textmap7 = Texture_Map(gray, windows[2]) 
    textimg3 = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    textimg5 = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    textimg7 = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    textimg3[:,:,0] = textmap3
    textimg5[:,:,1] = textmap5
    textimg7[:,:,2] = textmap7
    tmask3 = Otsu(textimg3, refines[0])
    tmask5 = Otsu(textimg5, refines[1])
    tmask7 = Otsu(textimg7, refines[2])
    
    return tmask3, tmask5, tmask7

def Texture_Map(img, N):
    grayimg = np.copy(img)
    grayimg = grayimg.astype(np.float64)
    pad = math.floor(N/2)
    height = grayimg.shape[0]
    width = grayimg.shape[1]
    textmap = np.zeros((height,width))
    for rows in range(pad, height-pad):
        for cols in range(pad, width-pad):
            box = grayimg[rows-pad:rows+pad+1, cols-pad:cols+pad+1]
            var = np.var(box)
            textmap[rows,cols] = var  
    # Normalize variance values and quantize to 255 gray levels
    textmap = (255*(textmap / textmap.max()))  
    textmap = textmap.astype(np.uint8)
    return textmap        
 
def Contour_Extraction(seg_img):
    height = seg_img.shape[0]
    width = seg_img.shape[1]
    contour = np.copy(seg_img)
    pad = 1
    # 8-neighbors contour evaluation
    for rows in range(pad, height-pad):
        for cols in range(pad, width-pad):
            if seg_img[rows,cols,0] == 255:
                neighbor8 = seg_img[rows-pad:rows+pad+1, cols-pad:cols+pad+1]
                if np.all(neighbor8):
                    contour[rows,cols] = 0
                else:
                    contour[rows,cols] = 255
    return contour

def main():
    path = 'C:/Users/user/Desktop/ECE661/HW6/images'
    filename = input("Input Image: ")
    segmentation = input("Segmentation Method: ")
    segname = input("Segmentation Output Name: ")
    cntrname = input("Contour Output Name: ")
    img = cv2.imread(os.path.join(path,filename))
    
    """ RGB Segmentation """
    if segmentation == 'RGB':
        if filename == 'baby.jpg':
            refines = [0,0,0]   # Number of iterations for k of each color channel
            blue_mask, green_mask, red_mask = RGB_Segmentation(img, refines)
            combine_mask = ~blue_mask & ~green_mask & red_mask # red dominant object in image
        elif filename == 'ski.jpg':
            refines = [0,0,0]
            blue_mask, green_mask, red_mask = RGB_Segmentation(img, refines)
            combine_mask = ~blue_mask & ~green_mask & red_mask
        elif filename == 'lighthouse.jpg':
            refines = [0,0,1]
            blue_mask, green_mask, red_mask = RGB_Segmentation(img, refines)
            combine_mask = ~blue_mask & ~green_mask & red_mask
        else:
            print("File does not exist in current directory")
            sys.exit()
            
        seg_image = np.zeros((combine_mask.shape[0],combine_mask.shape[1],3), dtype = np.uint8)
        for cols in range(0,combine_mask.shape[0]):
            for rows in range(0, combine_mask.shape[1]):
                if combine_mask[cols,rows] == 255:
                    seg_image[cols,rows] = 255
        path = 'C:/Users/user/Desktop/ECE661/HW6/outputs'
        cv2.imwrite(os.path.join(path,segname), seg_image)
                    
    """ Texture Segmentation """                
    if segmentation == 'Texture':
        windows = [3,5,7]
        if filename == 'baby.jpg':
            refines = [0,1,0]   # Number of iterations for k of each texture N
            mask_3, mask_5, mask_7 = Texture_Segmentation(img, windows , refines)
            combine_mask = mask_5  # Mask_5 preserves much more details after segmentation
        elif filename == 'ski.jpg':
            refines = [0,0,0]
            mask_3, mask_5, mask_7 = Texture_Segmentation(img, windows , refines)
            combine_mask = mask_3 & mask_5 & mask_7
        elif filename == 'lighthouse.jpg':
            refines = [0,0,0]
            mask_3, mask_5, mask_7 = Texture_Segmentation(img, windows , refines)
            combine_mask = mask_3 & mask_5 & mask_7
        else:
            print("File does not exist in current directory")
            sys.exit()
            
        seg_image = np.zeros((combine_mask.shape[0],combine_mask.shape[1],3), dtype = np.uint8)
        for cols in range(0,combine_mask.shape[0]):
            for rows in range(0, combine_mask.shape[1]):
                if combine_mask[cols,rows] == 255:
                    seg_image[cols,rows] = 255
        path = 'C:/Users/user/Desktop/ECE661/HW6/outputs'
        cv2.imwrite(os.path.join(path,segname),seg_image)     
    
    """ Contour Extraction """
    contour_img = Contour_Extraction(seg_image) 
    path = 'C:/Users/user/Desktop/ECE661/HW6/outputs'
    cv2.imwrite(os.path.join(path, cntrname), contour_img)
      
main()