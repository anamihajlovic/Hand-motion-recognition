# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread
from skimage.morphology import disk, diamond
from skimage.morphology import opening, closing, dilation
import os


#pretvori u sivo
def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)
    img_gray = 0*img_rgb[:, :, 0] + 0*img_rgb[:, :, 1] + 1*img_rgb[:, :, 2]  #crvene elemente tj krugove pretvaramo u crno, a plavu boju u belo
    img_gray = img_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return img_gray    

i = 0
for x in range(1):

    projectRoot = os.path.abspath(os.path.dirname(__file__))
    imgPath = os.path.join(projectRoot, 'TrainSet', 'palac0.jpg')
  
    img = imread(imgPath)
    img_gray = my_rgb2gray(img)   
    
    blurB = cv2.bilateralFilter(img_gray,9,150,150)      
    ret2,img_th = cv2.threshold(blurB, 110, 10, cv2.ADAPTIVE_THRESH_MEAN_C);    

    struct_elem = disk(5)
    img_dil = dilation(img_th, struct_elem)
    
    img_inv = 255 - img_dil
    image, contours, hiearchy = cv2.findContours(img_dil.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    plt.imshow(img_dil,'gray')
    contArea = cv2.contourArea(contours[0])           
    i += 1
    