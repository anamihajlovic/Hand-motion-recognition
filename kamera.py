# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:26:45 2017

@author: JELENA
@author: Ana
"""

import cv2
import numpy as np
import math
from skimage.morphology import diamond, square, disk
from skimage.morphology import opening, closing, dilation, erosion
import win32api
from testObrada import readModel
import matplotlib.pyplot as plt



#vraca se redosled boja BGR
def my_rgb2gray(frame_rgb):
    frame_gray = np.ndarray((frame_rgb.shape[0], frame_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)    
    frame_gray = 0*frame_rgb[:, :, 0] + 0*frame_rgb[:, :, 1] + 1*frame_rgb[:, :, 2]  #izdvajamo plave elemente na slici
    frame_gray = frame_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return frame_gray

#obrda frejma uz pomoc hsv color modela
def my_rgb2hsv(frame_rgb):
    frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    
    lower_pink = np.array([130, 100, 100])
    upper_pink = np.array([170, 255, 255])
    
    mask = cv2.inRange(frame_hsv, lower_pink, upper_pink)
    
    return mask

#podesava se pozicija kursora misa
def my_moveMouse(newX, newY):    
    currentX, currentY = win32api.GetCursorPos()  

    x = newX - currentX
    y = newY - currentY   
    
    win32api.SetCursorPos((currentX + x, currentY + y))
   
#frame sa kamere se obradjuje tako da se moze poslati kao ulaz u neuronsku mrezu
def createInputImage(frame, max_cont) :
    x,y,w,h = cv2.boundingRect(max_cont)        
    crop_img = frame.copy()[y: y+h, x:x+w]
    resized_frame = cv2.resize(crop_img, (50,50), interpolation = cv2.INTER_NEAREST)      
 
    img_array[0, :] = resized_frame.flatten()
    img_array[img_array > 0] = 10   
    data = np.zeros((1, 25), np.double)     
    for k in range(25):
        data[0, k] = np.mean(img_array[0, k*100:100*(k+1)])    
    return data

def my_Predict(model, img_input):        
    t = loaded_model.predict(img_input, verbose = 1)
    result = t.argmax(axis=1)
    print t[0][result], result[0]

    
def compareNumOfFingers(old_num, new_num):
    
    print("cmp" + str(old_num) + " " + str(new_num))
    return old_num == new_num
    
#funckija koja broji prste na frejmu
def countFingers(frame_cnt):
    finger_count = 0                    
    hulls = cv2.convexHull(max_cont,returnPoints = True) 
   
    max_distance = findMaxCenterDistance(hulls)
    big_distance = math.ceil(max_distance / 7)  
    small_distance = math.ceil(max_distance / 16)      
 
    k = 0               
    for k in range(hulls.shape[0]): 
         if(k == hulls.shape[0]-1):            
             x1, y1 = hulls[k-1,0]
             x2, y2 = hulls[k,0]
                
             hull_distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))          
                                      
             if(hull_distance > small_distance and y1 <= centerY):
                 if(checkCenterPosition(x1, y1, max_distance)):
                     finger_count += 1
                     hull_tuple = tuple(hulls[k][0])                                   
                     cv2.circle(frame_open,hull_tuple,10,[255,0,0],0)   
                                        
         else :            
             x1, y1 = hulls[k,0]
             x2, y2 = hulls[k+1,0]                  
                        
             hull_distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))          
                                      
             if(hull_distance > big_distance and y1 <= centerY):  
                 if(checkCenterPosition(x1, y1, max_distance)):
                     finger_count += 1
                     hull_tuple = tuple(hulls[k][0])                                   
                     cv2.circle(frame_open,hull_tuple,10,[255,0,0],0)                  
             
    return finger_count

#vraca se najvece rastojanje ispupcenja od centra konture
def findMaxCenterDistance(hulls):
    max_center_distance = -1
    k = 0
    for k in range(hulls.shape[0]): 
        x1, y1 = hulls[k,0]
        
    #udaljenost tacke od centra sake
        center_distance = math.sqrt((centerX-x1)*(centerX-x1) + (centerY-y1)*(centerY-y1))
        if(center_distance > max_center_distance):
            max_center_distance = center_distance  
     
    print("distance" + str(max_center_distance)) 
    return max_center_distance



#funkcija koja gleda poziciju pronadjenih tacaka, ukoliko nije odgovarajuca potrebno je preskociti tacku
def checkCenterPosition(x1, y1, max_distance):
    
    #if(max_distance >= 200):
     #   y_value = 100
    #else:
     #   y_value = 10
        
    if(y1 <= (centerY + 100) and y1 >= (centerY - 100)):
        if(x1 <= (centerX + 80) and x1 >= (centerX - 80)):        
            return False
       
    return True

 





print("Krenuo")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

loaded_model = readModel('model3.json', "model3.h5")   
#win32api.SetCursorPos(( win32api.GetSystemMetrics (0) / 2, win32api.GetSystemMetrics (1) / 2))
img_array =  np.zeros((1, 50*50), np.uint8)

frame_counter = 0
num_of_fingers = 0

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read() 
    frame_counter += 1    
    
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
    ret2,frame_th = cv2.threshold(blurB, 250, 0, cv2.THRESH_TOZERO);

    struct_elem = diamond(4)
    frame_open = opening(frame_th, struct_elem)
      
    proba = my_rgb2hsv(frame)
    proba_er = erosion(proba, disk(4))
    proba_open = dilation(proba_er, diamond(2))
    
    #proba_open = opening(proba, disk(4))
    
    # cv2.Canny(frame_open, 100, 100*3, frame_open, 3, True)
    image, contours, hiearchy = cv2.findContours(frame_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    max_area = 100
    cont_index = 0
   
    if(len(contours) != 0) :   
        old_frame_cnt = contours[0]
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
                                    
        #my_moveMouse(centerX * ratioX, centerY * ratioY)  
        
        
        if (frame_counter == 15):
            new_num = countFingers(max_cont)
            print("15new_num: " + str(new_num))
            if (num_of_fingers != new_num):  #predict radi samo kad ima razlike
                print("predict iz frame15")
                img_input = createInputImage(frame_open, max_cont)               
                my_Predict(loaded_model, img_input)
            
            num_of_fingers = new_num
            print("15num_of_fingers: " + str(num_of_fingers))
            

        elif (frame_counter == 30):
            new_num_of_fingers = countFingers(max_cont)
            print("30new_num of fingers: " + str(new_num_of_fingers))
            equal = compareNumOfFingers(num_of_fingers, new_num_of_fingers)
            frame_counter = 0
            if (not equal):
                print("predict iz frame30")
                img_input = createInputImage(frame_open, max_cont)               
                my_Predict(loaded_model, img_input)
            

    
   # print("counter: " + str(frame_counter))
    cv2.imshow("preview", frame_open)         
    rval, frame = vc.read()   
    frame_counter += 1
 
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break    
    if key == 32:          
        print("Poceo sam")   
        plt.imshow(frame_open, 'gray')
        print(centerX, centerY)
        #countFingers(max_cont, frame_open)                  
       #print(len(defects))
        #img_input = createInputImage(frame_open, max_cont)               
        #my_Predict(loaded_model, img_input)
        
    
vc.release()
cv2.destroyWindow("preview")