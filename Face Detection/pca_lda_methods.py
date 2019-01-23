# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:10:30 2018

@author: Hank Huang
"""
import numpy as np
import numpy.linalg as npla
import os
import cv2
import sys
import matplotlib.pyplot as plt

def Preprocess_Image(img_list, img_dir):
    labels = np.array([int(image[0:2]) for image in img_list])
    X = None
    # Vectorize each image
    for filename in img_list:
        image = cv2.imread(os.path.join(img_dir,filename))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        Xi = np.ravel(gray)
        Xi = np.reshape(Xi, (len(Xi),1))
        if X is not None:
            X = np.hstack((X, Xi))
        else:
            X = Xi
    # Compute global mean  
    X_mean = np.mean(X, axis = 1)
    img_vect = np.zeros((X.shape),dtype = float)
    # Normalize each image vector
    for idx in range(0,X.shape[1]):    
        img_vect[:,idx] = (X[:,idx] - X_mean) / npla.norm(X[:,idx] - X_mean)
    return img_vect, labels, X_mean

def PCA_Train(X):
    # Lower dimension trick for eigenvectors
    eigval, U = npla.eig(np.matmul(X.T, X))
    # Find K indices of largest eigenvalues based on mean
    eigval_l2s, U_l2s = EigenSort(eigval, U, 0)
    # Recover desired eigenvector in higher dimension
    W = np.matmul(X, U_l2s)
    return W

def LDA_Train(X, mean_global, total_class):
    N = X.shape[1] # total number of images
    class_size = int(N / total_class)

    mean_class = np.zeros((X.shape[0], total_class), dtype=float)
    class_diff = np.zeros((X.shape), dtype = float)
    class_idx = 0
    for i in range(0,N,class_size):
        mean_class[:,class_idx] = np.mean(X[:,i:i+class_size], axis = 1)
        mean_local = np.reshape(mean_class[:,class_idx], (X.shape[0],1))
        class_diff[:,i:i+class_size] = np.subtract(X[:,i:i+class_size] , mean_local)
        class_idx += 1
 
    mean_global = np.reshape(mean_global,(X.shape[0],1))
    mean_diff = np.subtract(mean_class, mean_global)
    
    """ Yu and Yang's Algorithm to ensure nonsingularity """
    # Lower dimension trick for eigenvectors
    eigval_Sb, eigvec_Sb = npla.eig(np.matmul(mean_diff.T, mean_diff)) 
    # Maximize between class scatter (descending order)
    valSb_l2s, vecSb_l2s = EigenSort(eigval_Sb, eigvec_Sb, 0)
    # Drop small eigenvalues
    keep_idx = np.where(valSb_l2s > 0)[0]
    eigval_drop = valSb_l2s[keep_idx]
    vecSb_l2s = vecSb_l2s[:,keep_idx]
    V = np.matmul(mean_diff, vecSb_l2s)
    M = V.shape[1] # First M eigenvectors in V
    
    # Yt * SB * Y = DB
    temp = 1 / np.sqrt(eigval_drop)
    DB_invsqrt = np.diag(temp[0:M])
    # Y : first M eigenvectors of V
    Y = V[:,0:M]
    Z = np.matmul(Y, DB_invsqrt)
    
    # Split computation in lower dimension instead of taking Sw directly
    Ztmean_diff = np.matmul(Z.T, mean_diff)
    ZtSwZ = np.matmul(Ztmean_diff, Ztmean_diff.T)
    DW, U = npla.eig(ZtSwZ)
    
    # Minimize within class scatter (ascending order)
    DW_s2l, U_s2l = EigenSort(DW, U, 1)
    W = np.matmul(Z, U_s2l)
    return W

def Classify(K, W, X_trn, X_tst, true_lbls):
    N = X_trn.shape[1]
    W = W[:,0:K]
    for i in range(0,W.shape[1]):
        W[:,i] = W[:,i] / npla.norm(W[:,i]) #Normalize eigenvectors
    # Project image vectors on K dim subspace as features
    y_train = np.zeros((K,N), dtype=float)
    for j in range(0,N): 
        y_train[:,j] = np.matmul(W.T, X_trn[:,j]) #Reduced dimension features to KxN
    
    predict = []
    for i in range(0,N):
        y_test = np.matmul(W.T, X_tst[:,i])
        y_test = np.reshape(y_test,(K,1))
        sq_dist = (np.subtract(y_train , y_test))**2
        euclidean = np.sqrt(np.sum(sq_dist,axis=0))
        neighbors = np.argsort(euclidean)
        NN = neighbors[0]
        predict.append(NN)
   
    temp = []
    for pred_idx in predict:
        temp.append(true_lbls[pred_idx])
    pred_lbls = np.array([temp])
      
    correct = len(np.nonzero(pred_lbls == true_lbls)[0])
    accuracy = (correct / N) * 100
    return accuracy
  
def EigenSort(value, vector, order):
    idx_sort = np.argsort(value)
    # Order: 1-> ascending, 0-> descending
    if order == 1: 
        value_sorted = value[idx_sort]
        vector_sorted = vector[:,idx_sort]
    else: 
        idx_sort = idx_sort[::-1]
        value_sorted = value[idx_sort]
        vector_sorted = vector[:,idx_sort]        
    return value_sorted, vector_sorted

def main():
    trndir = os.path.join(os.getcwd(),'train')
    tstdir = os.path.join(os.getcwd(),'test')
    train_list = os.listdir(trndir)
    test_list = os.listdir(tstdir)
    
    trn_vect, trn_lbls, mean_trn = Preprocess_Image(train_list, trndir)
    tst_vect, tst_lbls, mean_tst = Preprocess_Image(test_list, tstdir)
    
    total_class = 30 # Number of classes
    K = 30 # Number of reduced dimension feature vectors
    method = 'Compare'
    
    if method == 'PCA':
        W_PCA = PCA_Train(trn_vect)
        accuracy_PCA = Classify(K, W_PCA, trn_vect, tst_vect, trn_lbls)
        print("PCA Accuracy: %f%%" % accuracy_PCA)
    elif method == 'LDA':
        W_LDA = LDA_Train(trn_vect, mean_trn, total_class)
        accuracy_LDA = Classify(K, W_LDA, trn_vect, tst_vect, trn_lbls)
        print("LDA Accuracy: %f%%" % accuracy_LDA)    
    elif method == 'Compare':
        # Compare both performances and generate plot
        performance_PCA = []
        performance_LDA = []
        W_PCA = PCA_Train(trn_vect)
        W_LDA = LDA_Train(trn_vect, mean_trn, total_class)
        for Ki in range(0, K):
            accuracy_PCA = Classify(Ki, W_PCA, trn_vect, tst_vect, trn_lbls)
            performance_PCA.append(accuracy_PCA)
            
            accuracy_LDA = Classify(Ki, W_LDA, trn_vect, tst_vect, trn_lbls)
            performance_LDA.append(accuracy_LDA)
        krange = np.arange(1,K+1,1)
        plt.plot(krange, performance_PCA,'-xr',label='PCA')
        plt.plot(krange, performance_LDA, '-xb',label='LDA')
        plt.legend()
        plt.xticks(krange)
        #plt.plot(krange, performance_PCA,'-xr', krange, performance_LDA, '-xb')
        plt.xlabel('Subspace Dimension K')
        plt.ylabel('Accuracy')
        plt.title('Classifier Performance Comparison')
        plt.show()
    else:
        print("Error: No methods chosen")
        sys.exit()

main()


