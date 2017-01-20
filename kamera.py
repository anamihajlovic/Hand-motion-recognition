# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:26:45 2017

@author: JELENA
"""

import cv2
import numpy as np
from skimage.morphology import disk, square, diamond
from skimage.morphology import opening, closing


#vraca se redosled boja BGR
def my_rgb2gray(frame_rgb):
    frame_gray = np.ndarray((frame_rgb.shape[0], frame_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)    
    frame_gray = 0*frame_rgb[:, :, 0] + 1*frame_rgb[:, :, 1] + 0*frame_rgb[:, :, 2]  #izdvajamo plave elemente na slici
    frame_gray = frame_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return frame_gray


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = my_rgb2gray(frame)
             
    blurB = cv2.bilateralFilter(frame_gray,9,150,150)    
    ret2,frame_th = cv2.threshold(blurB, 250, 0, cv2.THRESH_TOZERO);

    struct_elem = diamond(4)
    frame_open = closing(frame_th, struct_elem)
    
    cv2.imshow("preview", frame_th)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
vc.release()
cv2.destroyWindow("preview")
