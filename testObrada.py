# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread
from skimage.morphology import disk, diamond, square
from skimage.morphology import opening, closing, dilation
import os


#RAZMISLITI O JOS NEKIM MORFOLOSKIM OPERACIJAMA

#pretvori u sivo
def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)
    img_gray = 0*img_rgb[:, :, 0] + 0*img_rgb[:, :, 1] + 1*img_rgb[:, :, 2]  #crvene elemente tj krugove pretvaramo u crno, a plavu boju u belo
    img_gray = img_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return img_gray    

i = 0
for x in range(1):

    projectRoot = os.path.abspath(os.path.dirname(__file__))
    imgPath = os.path.join(projectRoot, 'TrainSet', 'saka9.jpg')
  
    img = imread(imgPath)
    img_gray = my_rgb2gray(img)   
    
    blurB = cv2.bilateralFilter(img_gray,9,150,150)      
    ret1,img_th = cv2.threshold(blurB, 110, 10, cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV);       

    struct_elem = disk(5)
    img_dil = dilation(img_th, struct_elem)        
                     
    #deluje mi da malo prosiri beline, ali uspori program, RAZMISLITI O TOME!
    #elem = disk(10)
    #img_open = opening(img_dil, elem)

    #findContours -> belo mu je foreground a crno background   
    image, contours, hiearchy = cv2.findContours(img_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 2000
    cont_index = 0        
   
    for i in range(len(contours)) :
        c = contours[i]
        area = cv2.contourArea(c)
        if(area > max_area) :                      
            max_area = area
            cont_index = i
                    
        max_cont = contours[cont_index] 

    x,y,w,h = cv2.boundingRect(max_cont)        
    crop_img = img_th[y: y+h, x:x+w]
    #plt.imshow(img_inv, 'gray')
    plt.imshow(crop_img, 'gray') 

     