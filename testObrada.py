# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread
from skimage.morphology import disk
from skimage.morphology import opening, closing, dilation
import os
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.models import model_from_json
#RAZMISLITI O JOS NEKIM MORFOLOSKIM OPERACIJAMA


#pretvori u sivo
def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)
    img_gray = 0*img_rgb[:, :, 0] + 0*img_rgb[:, :, 1] + 1*img_rgb[:, :, 2]  #crvene elemente tj krugove pretvaramo u crno, a plavu boju u belo
    img_gray = img_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return img_gray    
    
#kreiranje skupa izlaza iz neuronske mreze
def createTrainOut(row_num, gest_num):
    trainOut = np.zeros((row_num, gest_num), np.uint8)
    trainOut[:17, 0] = 1
    trainOut[17:31, 2] = 1
    trainOut[31:45, 3] = 1
    trainOut[45:61, 1] = 1
    return trainOut

#kreiranje modela neuronske mreze (bez skrivenih slojeva) i cuvanje u fajl
def createModel(data, trainOut, gestures):    
    model = Sequential()
    model.add(Dense(10, input_dim=25))
    model.add(Activation('sigmoid'))
    model.add(Dense(len(gestures)))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=0.00001, momentum=0.7)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    training = model.fit(data, trainOut, nb_epoch=5000, batch_size=20, verbose=0)
    print training.history['loss'][-1]

    # evaluacija modela
    scores = model.evaluate(data, trainOut, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # cuvanje modela u json fajl
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # cuvanje tezina u HDF5 fajl
    model.save_weights("model.h5")
    print("Model je sacuvan u fajl")

#kreiranje modela neuronske mreze (sa jednim skrivenim slojem) i cuvanje u fajl
def createModel2(data, trainOut, gestures):    
    model = Sequential()
    model.add(Dense(15, input_dim=25))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, input_dim = 15))
    model.add(Activation('sigmoid'))
    model.add(Dense(len(gestures)))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=0.00001, momentum=0.7)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    training = model.fit(data, trainOut, nb_epoch=5000, batch_size=20, verbose=0)
    print training.history['loss'][-1]

    # evaluacija modela
    scores = model.evaluate(data, trainOut, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # cuvanje modela u json fajl
    model_json = model.to_json()
    with open("model2.json", "w") as json_file:
        json_file.write(model_json)
    # cuvanje tezina u HDF5 fajl
    model.save_weights("model2.h5")
    print("Model je sacuvan u fajl")
    
#kreiranje modela neuronske mreze (sa jednim skrivenim slojem), promenjene su vrednosti sgd i cuvanje u fajl
def createModel3(data, trainOut, gestures):    
    model = Sequential()
    model.add(Dense(15, input_dim=25))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, input_dim = 15))
    model.add(Activation('sigmoid'))
    model.add(Dense(len(gestures)))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=0.000001, momentum=0.5)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    training = model.fit(data, trainOut, nb_epoch=5000, batch_size=20, verbose=0)
    print training.history['loss'][-1]

    # evaluacija modela
    scores = model.evaluate(data, trainOut, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # cuvanje modela u json fajl
    model_json = model.to_json()
    with open("model3.json", "w") as json_file:
        json_file.write(model_json)
    # cuvanje tezina u HDF5 fajl
    model.save_weights("model3.h5")
    print("Model je sacuvan u fajl")


# ucitavanje modela iz fajla i njegovo kreiranje
def readModel(json_file, h5_file):   
    json_file = open(json_file, 'r')
   # json_file = open('model3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # ucitavanje tezina u kreirani model
    loaded_model.load_weights(h5_file)
    #loaded_model.load_weights("model3.h5")
    print("Ucitan model iz fajla")

    # evaluacija ucitanog modela
    sgd = SGD(lr=0.1, decay=0.00001, momentum=0.7)
    loaded_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
   # score = loaded_model.evaluate(data, trainOut, verbose=0)
   # print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)
    return loaded_model

#obrada slika iz train seta za slanje neuronskoj mrezi
def mainFunction():
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
        img_open = opening(img_th, struct_elem)        
      
        #deluje mi da malo smanji beline, ali uspori program, RAZMISLITI O TOME!
        #elem = disk(8)
        #img_open = closing(img_th, elem)   

        #findContours -> belo mu je foreground a crno background   
        image, contours, hiearchy = cv2.findContours(img_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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
        crop_img = img_open.copy()[y: y+h, x:x+w]
        #cv2.drawContours(img_th, contours, cont_index, (255,0,0), 5)
        #plt.imshow(img_th, 'gray')
        #plt.imshow(crop_img, 'gray')
    
        resized_img = cv2.resize(crop_img, (50,50), interpolation = cv2.INTER_NEAREST)
        img_array[i, :] = resized_img.flatten()
  
    data = np.zeros((img_num, 25), np.double)
    row_array = np.zeros((1,25), np.double)
    for i in range(img_num):        
        for k in range(25):
            row_array[0, k] = np.mean(img_array[i, k*100:100*(k+1)])
        
        data[i, :] = row_array

    trainOut = createTrainOut(img_num, len(gestures))    
    #createModel(data, trainOut, gestures)
    #createModel3(data, trainOut, gestures)
    
    

