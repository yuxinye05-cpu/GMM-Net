import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy


def preprocessed_image_show(train_imgs_0,train_imgs_1,train_imgs_2,train_imgs_3,train_imgs_4,save_path):
     train_imgs_0=np.transpose(train_imgs_0[2],(1,2,0))
     train_imgs_1=np.repeat((np.transpose(train_imgs_1[2],(1,2,0))).astype(np.uint8),repeats=3,axis=2)
     train_imgs_2=np.repeat((np.transpose(train_imgs_2[2],(1,2,0))).astype(np.uint8),repeats=3,axis=2)
     train_imgs_3 = np.repeat((np.transpose(train_imgs_3[2], (1, 2, 0))).astype(np.uint8), repeats=3, axis=2)
     train_imgs_4 = np.repeat((np.transpose(train_imgs_4[2], (1, 2, 0))).astype(np.uint8), repeats=3, axis=2)

     total_img = Image.fromarray(np.concatenate((train_imgs_0, train_imgs_1, train_imgs_2, train_imgs_3,train_imgs_4), axis=1).astype(np.uint8))
     save_path_img = save_path + '/preprocessed_image_show.png'
     total_img.save(save_path_img)


#group a set of img patches 
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg  #ndarray:(320,640,1) 将50张图片拼成一张大图

# Prediction result splicing (original img, predicted probability, binary img, groundtruth)
def concat_result(ori_img,pred_res,gt):
    ori_img = data = np.transpose(ori_img,(1,2,0))
    pred_res = data = np.transpose(pred_res,(1,2,0))
    gt = data = np.transpose(gt,(1,2,0))

    binary = deepcopy(pred_res)
    binary[binary>=0.5]=1
    binary[binary<0.5]=0  

    if ori_img.shape[2]==3:
        pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
        binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
        gt = np.repeat((gt*255).astype(np.uint8),repeats=3,axis=2)
    total_img = np.concatenate((ori_img,pred_res,binary,gt),axis=1)
    return total_img

#visualize image, save as PIL image
def save_img(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    img = Image.fromarray(data.astype(np.uint8))  #the image is between 0-1
    img.save(filename)
    return img