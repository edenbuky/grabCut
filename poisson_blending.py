import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    src,mask = newSrcMask(im_src, im_tgt, im_mask, center)
    #im_blend = src[center[1]-len(im_src)//2:center[1]-len(im_src)//2+len(im_src)][center[0]-len(im_src[0])//2:center[0]-len(im_src[0])//2+len(im_src[0])]
    srcLap = Laplacian(src)
    A,b0,b1,b2 = computeAandB(srcLap,im_tgt,mask)
    f0,f1,f2 = getF(A,b0,b1,b2)
    calibrateF(f0,f1,f2)
    im_blend = fToImage(f0,f1,f2,im_tgt).astype(np.uint8)
    return im_blend

def mask_check(src,mask):
    check = np.zeros((len(src),len(src[0]),len(src[0][0])))
    for i in range(len(src)):
        for j in range(len(src[0])):
            if mask[i][j] != 0:
                check[i][j][0] = src[i][j][0]
                check[i][j][1] = src[i][j][1]
                check[i][j][2] = src[i][j][2]
    return check

def calibrateF(f0,f1,f2):
    for i in range(len(f0)):
        if f0[i]>255:
            f0[i] = 255
        if f0[i]<0:
            f0[i]=0
        if f1[i]>255:
            f1[i] = 255
        if f1[i]<0:
            f1[i]=0
        if f2[i]>255:
            f2[i] = 255
        if f2[i]<0:
            f2[i]=0
def calibrateF2(f0,f1,f2):
    f0max = f0.max()
    f0min = f0.min()
    f1max = f1.max()
    f1min = f1.min()
    f2max = f2.max()
    f2min = f2.min()
    deltaf0 = f0max-f0min
    deltaf1 = f1max-f1min
    deltaf2 = f2max-f2min
    for i in range(len(f0)):
        f0[i] = (f0[i]-f0min) / deltaf0 * 255
        f1[i] = (f1[i]-f1min)/deltaf0 * 255
        f2[i] = (f2[i]-f2min)/deltaf0 * 255

def Laplacian(src):
    laplacian = np.zeros((len(src),len(src[0]),len(src[0][0])))
    for i in range(len(src)):
        for j in range(len(src[0])):
            if i>0:
                laplacian[i][j][0]+=src[i-1][j][0]
                laplacian[i][j][1]+=src[i-1][j][1]
                laplacian[i][j][2]+=src[i-1][j][2]
            if i<len(src)-1:
                laplacian[i][j][0]+=src[i+1][j][0]
                laplacian[i][j][1]+=src[i+1][j][1]
                laplacian[i][j][2]+=src[i+1][j][2]
            if j>0:
                laplacian[i][j][0]+=src[i][j-1][0]
                laplacian[i][j][1]+=src[i][j-1][1]
                laplacian[i][j][2]+=src[i][j-1][2]
            if j<len(src[0])-1:
                laplacian[i][j][0]+=src[i][j+1][0]
                laplacian[i][j][1]+=src[i][j+1][1]
                laplacian[i][j][2]+=src[i][j+1][2]
            laplacian[i][j][0]-=4*src[i][j][0]
            laplacian[i][j][1]-=4*src[i][j][1]
            laplacian[i][j][2]-=4*src[i][j][2]
    return laplacian

def newSrcMask(im_src, im_tgt, im_mask, center):
    newMask = np.zeros((len(im_tgt),len(im_tgt[0])))
    newSrc = np.zeros((len(im_tgt),len(im_tgt[0]),len(im_tgt[0][0])))
    for i in range(len(im_src)):
        #print(center[0]-len(im_src)//2+i)
        for j in range(len(im_src[0])):
            if center[1]-len(im_src)//2+i>=0 and center[1]-len(im_src)//2+i<len(newSrc) and center[0]-len(im_src[0])//2+j>=0 and center[0]-len(im_src[0])//2+j<len(newSrc[0]):
                newSrc[center[1]-len(im_src)//2+i][center[0]-len(im_src[0])//2+j][0] = im_src[i][j][0]
                newSrc[center[1]-len(im_src)//2+i][center[0]-len(im_src[0])//2+j][1] = im_src[i][j][1]
                newSrc[center[1]-len(im_src)//2+i][center[0]-len(im_src[0])//2+j][2] = im_src[i][j][2]
    for i in range(len(im_mask)):
        for j in range(len(im_mask[0])):
            if center[1]-len(im_mask)//2+i>=0 and center[1]-len(im_mask)//2+i<len(newSrc) and center[0]-len(im_mask[0])//2+j>=0 and center[0]-len(im_mask[0])//2+j<len(newSrc[0]):
                newMask[center[1]-len(im_mask)//2+i][center[0]-len(im_mask[0])//2+j] = im_mask[i][j]
    return newSrc,newMask


def computeAandB(srcLap,tgt,mask):
    m=len(tgt[0])
    n= len(tgt)
    A =  scipy.sparse.lil_matrix((m*n, m*n))
    b0=np.zeros(m*n)
    b1=np.zeros(m*n)
    b2=np.zeros(m*n)
    for i in range(m*n):
        if mask[i//m][i%m]==0:
            A[i,i]=1
            #print(len(b0))
            b0[i]=tgt[i//m][i%m][0]
            b1[i]=tgt[i//m][i%m][1]
            b2[i]=tgt[i//m][i%m][2]
        else:
            b0[i] = srcLap[i//m][i%m][0]
            b1[i] = srcLap[i//m][i%m][1]
            b2[i] = srcLap[i//m][i%m][2]
            A[i,i]=-4
            if i//m!=0:
                if mask[i//m-1][i%m]==0:
                    b0[i]-=tgt[i//m-1][i%m][0]
                    b1[i]-=tgt[i//m-1][i%m][1]
                    b2[i]-=tgt[i//m-1][i%m][2]
                else:
                    A[i,i-m]=1
            if i//m!=n-1:
                if mask[i//m+1][i%m]==0:
                    b0[i]-=tgt[i//m+1][i%m][0]
                    b1[i]-=tgt[i//m+1][i%m][1]
                    b2[i]-=tgt[i//m+1][i%m][2]
                else:
                    A[i,i+m]=1
            if i%m!=0:
                if mask[i//m][i%m-1]==0:
                    b0[i]-=tgt[i//m][i%m-1][0]
                    b1[i]-=tgt[i//m][i%m-1][1]
                    b2[i]-=tgt[i//m][i%m-1][2]
                else:
                    A[i,i-1]=1
            if i%m!=m-1:
                if mask[i//m][i%m+1]==0:
                    b0[i]-=tgt[i//m][i%m+1][0]
                    b1[i]-=tgt[i//m][i%m+1][1]
                    b2[i]-=tgt[i//m][i%m+1][2]
                else:
                    A[i,i+1]=1
    return A,b0,b1,b2


def getF(A,b0,b1,b2):
    #ransA = A.transpose()
    #transADotA = transA.dot(A)
    #inv = np.linalg.inv(transADotA.todense())
    #denseTransA = transA.todense()
    #f0 = inv.dot(denseTransA).dot(b0)
    #f1 = inv.dot(denseTransA).dot(b1)
    #f2 = inv.dot(denseTransA).dot(b2)
    A_scr = A.tocsr()
    f0 = spsolve(A_scr, b0)
    f1 = spsolve(A_scr, b1)
    f2 = spsolve(A_scr, b2)
    #print(f1.max())
    return f0,f1,f2

def fToImage(f0,f1,f2,im_tgt):
    res = np.zeros((len(im_tgt),len(im_tgt[0]),len(im_tgt[0][0])))
    for i in range(len(im_tgt)):
        for j in range(len(im_tgt[0])):
            #s = f0[i*im_tgt[0]+j]
            #print(f0.shape)
            #print(s)
            #print(len(s))
            #d = s[0]
            #print(type(d))
            
            res[i][j][0] = f0[i*len(im_tgt[0])+j]
            res[i][j][1] = f1[i*len(im_tgt[0])+j]
            res[i][j][2] = f2[i*len(im_tgt[0])+j]
    return res



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
