# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread
from skimage.morphology import disk
from skimage.morphology import opening, closing, dilation
import os
#RAZMISLITI O JOS NEKIM MORFOLOSKIM OPERACIJAMA


#pretvori u sivo
def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)
    img_gray = 0*img_rgb[:, :, 0] + 0*img_rgb[:, :, 1] + 1*img_rgb[:, :, 2]  #crvene elemente tj krugove pretvaramo u crno, a plavu boju u belo
    img_gray = img_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return img_gray    
    
def createTrainOut(row_num, gest_num):
    trainOut = np.zeros((row_num, gest_num), np.uint8)
    trainOut[:17, 0] = 1
    trainOut[17:31, 2] = 1
    trainOut[31:45, 3] = 1
    trainOut[45:61, 1] = 1
    return trainOut


gestures = ['two', 'three', 'thumb', 'fist']
labels = np.arange(0, len(gestures))

img_num = 61
img_array =  np.zeros((img_num, 50*50), np.uint8)
for i in range(img_num):
    
    projectRoot = os.path.abspath(os.path.dirname(__file__))
    imgPath = os.path.join(projectRoot, 'TrainSet', 'img' +  str(i) + '.jpg')
  
    #imgPath = "C:\Users\Olivera\Desktop\TestSet\im7.jpg"
   
    img = imread(imgPath)
    img_gray = my_rgb2gray(img)   
    
    blurB = cv2.bilateralFilter(img_gray,9,150,150)      
    ret1,img_th = cv2.threshold(blurB, 110, 10, cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV);       

    struct_elem = disk(5)
    img_dil = dilation(img_th, struct_elem)        
                     
    #deluje mi da malo smanji beline, ali uspori program, RAZMISLITI O TOME!
    #elem = disk(8)
    #img_open = closing(img_th, elem)   

    #findContours -> belo mu je foreground a crno background   
    image, contours, hiearchy = cv2.findContours(img_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 2000
    cont_index = 0        
   
    for x in range(len(contours)) :
        c = contours[x]
        area = cv2.contourArea(c)
        if(area > max_area) :                      
            max_area = area
            cont_index = x
                    
        max_cont = contours[cont_index] 

    x,y,w,h = cv2.boundingRect(max_cont)        
    crop_img = img_th.copy()[y: y+h, x:x+w]
    #cv2.drawContours(img_th, contours, cont_index, (255,0,0), 5)
    #plt.imshow(img_th, 'gray')
    #plt.imshow(crop_img, 'gray')
    
    resized_img = cv2.resize(crop_img, (50,50), interpolation = cv2.INTER_NEAREST)
    #plt.imshow(resized_img, 'gray')      
    
    img_array[i, :] = resized_img.flatten()
    #print(img_array[i,2])

data = np.zeros((img_num, 25), np.double)
row_array = np.zeros((1,25), np.double)
for i in range(img_num):        
    for k in range(25):
        row_array[0, k] = np.mean(img_array[i, k*100:100*(k+1)])
        
    data[i, :] = row_array

trainOut = createTrainOut(img_num, len(gestures))
