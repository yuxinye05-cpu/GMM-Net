###################################################
#
#   Script to pre-process the original imgs
#
##################################################

import numpy as np
import cv2


# 预处理图片展示
def preprocessed_image(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs_0 = data
    train_imgs = rgb2gray(data)
    train_imgs_1=train_imgs
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs_2 = train_imgs
    train_imgs = clahe_equalized(train_imgs)
    train_imgs_3 = train_imgs
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs_4 = train_imgs
    return train_imgs_0,train_imgs_1,train_imgs_2,train_imgs_3,train_imgs_4

#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs

#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


def FTSalience(imgs):
    # print("imgs.shape")
    # print(imgs.shape)
    FT_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        FT_img = np.array(imgs[i,0], dtype = np.uint8)
        # print("FT_img.shape")
        # print(FT_img.shape)
        img_mean = np.mean(FT_img)
        FT_img = cv2.GaussianBlur(FT_img, (5,5), 1.5, 1.5)
        FT_imgs[i,0] = abs(FT_img-0.2*img_mean)
    return FT_imgs

def ImgEnhance(imgs):
    # print("imgs.shape")
    # print(imgs.shape)
    Enhance_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        Src = np.array(imgs[i,0], dtype = np.uint8)
        # print("Src.shape")
        # print(Src.shape)
        srcGauss = cv2.GaussianBlur(Src, (5,5), 1.5, 1.5)
        cv2.Laplacian(srcGauss,-1, srcGauss, 3, 1.0, 1.0)
        #Enhance_imgs[i,0] = Src-srcGauss
        Enhance_imgs[i,0] = cv2.subtract(Src, srcGauss)
    return Enhance_imgs

