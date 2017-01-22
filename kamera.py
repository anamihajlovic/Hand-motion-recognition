# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:26:45 2017

@author: JELENA
@author: Ana
"""

import cv2
import numpy as np
import math
from skimage.morphology import diamond
from skimage.morphology import opening, closing
import win32api, win32con


#vraca se redosled boja BGR
def my_rgb2gray(frame_rgb):
    frame_gray = np.ndarray((frame_rgb.shape[0], frame_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)    
    frame_gray = 0*frame_rgb[:, :, 0] + 0*frame_rgb[:, :, 1] + 1*frame_rgb[:, :, 2]  #izdvajamo plave elemente na slici
    frame_gray = frame_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return frame_gray

#podesava se pozicija kursora misa
def my_moveMouse(newX, newY):    
    currentX, currentY = win32api.GetCursorPos()  

    x = newX - currentX
    y = newY - currentY   
    
    win32api.SetCursorPos((currentX + x, currentY + y))
   

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

win32api.SetCursorPos(( win32api.GetSystemMetrics (0) / 2, win32api.GetSystemMetrics (1) / 2))

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()     
    
    frame_width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if(frame_width != 0 and frame_height != 0) :
        ratioX = (int) (math.ceil(win32api.GetSystemMetrics (0) / frame_width))        
        ratioY = (int) (math.ceil(win32api.GetSystemMetrics (1) / frame_height))
    
else:
    rval = False

while rval:
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = my_rgb2gray(frame)
             
    blurB = cv2.bilateralFilter(frame_gray,9,150,150)    
    ret2,frame_th = cv2.threshold(blurB, 240, 0, cv2.THRESH_TOZERO);

    struct_elem = diamond(4)
    frame_open = opening(frame_th, struct_elem)        
    
    # cv2.Canny(frame_open, 100, 100*3, frame_open, 3, True)
    
    image, contours, hiearchy = cv2.findContours(frame_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    max_area = 100
    cont_index = 0

    if(len(contours) != 0) :    
        for i in range(len(contours)) :
            c = contours[i]
            area = cv2.contourArea(c)
            if(area > max_area) :
                max_area = area
                cont_index = i
                    
        max_cont = contours[cont_index]     
        M = cv2.moments(max_cont)         
            
        if(M['m00'] != 0) :
            centerX = int(M['m10']/M['m00'])
            centerY = int(M['m01']/M['m00'])        
                                    
        my_moveMouse(centerX * ratioX, centerY * ratioY)
 
    cv2.imshow("preview", frame_open)    
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    
vc.release()
cv2.destroyWindow("preview")