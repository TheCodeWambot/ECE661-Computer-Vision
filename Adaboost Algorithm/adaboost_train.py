# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:51:02 2018

@author: Hank Huang
"""
import numpy as np
import numpy.linalg as npla
import os
import cv2
import sys
import math

def LoadTrainImg(imgdir):
    file_list = os.listdir(imgdir)
    imglist = []
    for filename in file_list:
        image = cv2.imread(os.path.join(imgdir,filename))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(float)
        imglist.append(gray)     
    return imglist

def Convert2Integral(img_dim, pad, images, idx):
    # Add top and left zero paddings for integrals at image boundaries
    h = img_dim[1] + pad
    w = img_dim[0] + pad
    int_img = np.zeros((h, w),dtype = float)
    int_img[1:h,1:w] = np.cumsum(np.cumsum(images[idx], axis=1), axis=0)
    return int_img

def SumPixels(int_img, region):
    row = region[0]
    col = region[1]
    x = region[2]
    y = region[3]
    boxA = int_img[row, col]
    boxB = int_img[row, col+x]
    boxC = int_img[row+y, col]
    boxD = int_img[row+y, col+x]
    regionSum = boxA + boxD - (boxB + boxC)
    return regionSum

def HaarFeatures(width, height, pad, int_img):
    hpad = height + pad
    wpad = width + pad
    feature = []
    # Horizontal Features 
    #for y in range(1, height+1):
    for x in range(1, int(width/2)+1):
        # Raster Scan with single haar feature
        for rows in range(0, int(hpad-1)):
            for cols in range(0, int(wpad - 2*x)):
                whiteregion = [rows,cols,x,1]
                darkregion = [rows,cols+x,x,1]
                diff = SumPixels(int_img, darkregion) - SumPixels(int_img, whiteregion)
                feature.append(diff)
    
    # Vertical Features                 
    for y in range(1, int(height/2)+1):
        #for x in range(1, width+1):
        # Raster Scan with single haar feature
        for rows in range(0, int(hpad - 2*y)):
            for cols in range(0, int(wpad - 2)):
                whiteregion = [rows,cols,2,y]
                darkregion = [rows+y,cols,2,y]
                diff = SumPixels(int_img, darkregion) - SumPixels(int_img, whiteregion)
                feature.append(diff)
    
    return feature

def FeatureExtraction(featsize, images, img_dim):
    datasize = len(images)
    features = np.zeros((featsize, datasize),dtype=float)
    pad = 1
    # Convert to integral image and extract 166000 features each
    for idx in range(0, datasize):
        int_img = Convert2Integral(img_dim, pad, images, idx)
        features[:,idx] = HaarFeatures(img_dim[0], img_dim[1], pad, int_img)
        #print(idx)
    return features

""" Get Best Weak Classifier """
def BWClassifier(features, weight, labels, totPos):
    global theta,best_weak_idx,best_min_err,best_polarity, best_weak_pred
    
    # Normalize weights
    weight = np.divide(weight, np.sum(weight))
    featsize = features.shape[0]
    totImgs = features.shape[1]
    best_min_err = math.inf 
    
    TPos = np.full((totImgs,1),np.sum(weight[0:totPos,0]))
    TNeg = np.full((totImgs,1),np.sum(weight[totPos:totImgs,0]))

    for f in range(0, featsize):
        singlefeat = features[f,:]
        feat_sort = np.sort(singlefeat)
        idx_sort = np.argsort(singlefeat)
        weight_sort = weight[idx_sort,0]
        label_sort = labels[idx_sort,0]
    
        SPos = np.cumsum(np.multiply(weight_sort,label_sort), axis=0)
        SNeg = np.subtract(np.cumsum(weight_sort, axis=0) , SPos)
        err_lblPos = SPos + (TNeg.T - SNeg)
        err_lblNeg = SNeg + (TPos.T - SPos)       
        misclass_errs = np.minimum(err_lblPos, err_lblNeg)
        min_err = np.amin(misclass_errs)
        thresh_idx = np.argmin(misclass_errs)
        
    # Weak classifier image prediction by threshold
        weak_pred = np.zeros((totImgs,1))
        if err_lblPos[0][thresh_idx] <= err_lblNeg[0][thresh_idx]:
            # polarity -1: below threshold object detected
            p = -1
            weak_pred[thresh_idx:totImgs] = 1
            weak_pred[idx_sort] = weak_pred
        else:
            # polartiy 1: above threshold object detected
            p = 1
            weak_pred[0:thresh_idx] = 1
            weak_pred[idx_sort] = weak_pred
     # Update for best weak classifier parameters by minimum error
        if min_err < best_min_err:
            best_min_err = min_err
            best_weak_pred = weak_pred
            best_weak_idx = f
            best_polarity = p
            offset = 0.5 # Offset for cases where all features above/below threshold
            if thresh_idx == 0:
                theta = feat_sort[0] - offset
            elif thresh_idx == totImgs-1:
                theta = feat_sort[totImgs-1] + offset
            else:
                # Most cases, set threshold to mean of features
                theta = (feat_sort[thresh_idx] + feat_sort[thresh_idx])/2
    print(best_min_err)       
    # Best parameters for one weak classifier     
    best_htparams = np.array([theta, best_polarity, best_weak_idx, best_min_err])             
    return best_htparams, best_weak_pred
            
def CascadeTrain(stage, features, T, tTPR, tFPR, totPos):
    totImgs = features.shape[1]
    print("Total Images: %d" % totImgs)
    totNeg = totImgs - totPos
    weight = np.zeros((totImgs,1), dtype=float)
    labels = np.zeros((totImgs,1), dtype=float)
    
    # Initialize weights to 1/2*Pos and 1/2*Neg from Viola and Jones
    # Label 1 for positive and 0 for negative images
    weight[0:totPos,0] = 0.5 / totPos
    labels[0:totPos,0] = 1
    weight[totPos:totImgs,0] = 0.5 / totNeg
    alpha = np.zeros((T,1), dtype=float)
    ht = np.zeros((4,T),dtype=float)
    #strong_accumulate = np.zeros((totImgs, T),dtype=float)
    weak_classify = np.zeros((totImgs,T))
    
    """ Adaboost Routine """
    for t in range(0,T):
        # Compute weak classifier 
        ht_params, ht_pred = BWClassifier(features, weight, labels, totPos)
        ht[:,t] = ht_params
        weak_classify[:,t] = np.reshape(ht_pred, (totImgs,))
        error = ht_params[3]
        # Obtain trust value for each ht
        alpha[t,0] = np.log((1-error)/error)  
        # Update weights for each image
        weight = np.multiply(weight, np.power((error/(1-error)), 1-np.logical_xor(labels,ht_pred)))
        strong_pred = np.matmul(weak_classify[:,0:t+1], alpha[0:t+1,0])
        thresh = np.amin(strong_pred[0:totPos]) # We want 100% TP rate
        
        # Stronger prediction with current accumulated weak classifiers
        strong_result = []
        for i in range(0,totImgs):
            if strong_pred[i] >= thresh:
                strong_result.append(1)
            else:
                strong_result.append(0)
        
        TPR = np.sum(strong_result[0:totPos]) / totPos
        FPR = np.sum(strong_result[totPos:totImgs]) / totNeg
         
        if TPR == tTPR and FPR <= tFPR:
            print("Stage %d weak classifiers: %d" % (stage,(t+1)))
            print("TPR = %f" % TPR)
            print("FPR = %f" % FPR)
            break
       
    # Keep negative images that are misclassified as positive
    neg_result = strong_result[totPos:totImgs]
    neg_sort = np.sort(neg_result)
    keep_idx = np.argsort(neg_result)
    for j in range(0,totNeg):
        if neg_sort[j] > 0:
            keep_idx = keep_idx[j:totNeg]
            break    
    
    total_ht = t+1
    stage_params = np.array([ht, alpha, thresh, total_ht]) 
    return stage_params,keep_idx

            
def main():
    trndir = os.path.join(os.getcwd(),'train')
    Pos_dir = os.path.join(trndir,'positive')
    Neg_dir = os.path.join(trndir,'negative')
    
    pos_images = LoadTrainImg(Pos_dir)
    neg_images = LoadTrainImg(Neg_dir)
    all_images = pos_images + neg_images
    
    width = 40  # Single fixed size for all images
    height = 20
    img_dims = np.array([width, height])
      
    if os.path.isfile('haar_featuresV4.npy'):
        train_features = np.load('haar_featuresV4.npy')
        print(train_features.shape)
    else:
        total_features = 11900
        train_features = FeatureExtraction(total_features, all_images, img_dims)
        np.save('haar_featuresV4', train_features)
    
    stages = 15
    T = 100 # Weak classifiers each stage
    tTPR = 1 # Target true positive rate
    tFPR = 0.6 # Target false positive rate
    pos_size = len(pos_images)
    train_params = []
    
    for s in range(0,stages):
        stage_params, keep_idx = CascadeTrain(s, train_features, T, tTPR, tFPR, pos_size)
        pos_keep = train_features[:,0:pos_size]
        print("Negative size: %d" % len(keep_idx))
        neg_keep = train_features[:,pos_size+keep_idx]
        
        train_features = np.hstack((pos_keep,neg_keep))       
        train_params.append(stage_params)
        if train_features.shape[1] == pos_size:
            break
    
    # Store training parameters for testing    
    if os.path.isfile('train_params.npy'):
        train_params = np.load('train_params.npy')
    else:
        np.save('train_params', train_params)    
main() 

