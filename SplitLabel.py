

from distutils.archive_util import make_archive
from os import wait
from time import sleep
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import functools

def Split(src):
    h, w, c = src.shape
    

    #src = np.uint8(np.clip((255/src.max() * src ), 0, 255))

    cv2.imwrite('Matricula.png',src)

    scale = np.average(src)/128
    print(scale)
    
    blurred = cv2.medianBlur(src, 5)   

    for i in range(0,10):

        characters = []
         
        OnlyBlack = cv2.inRange(blurred,(0,0,0),(pow(scale,2)*100+(15*i),pow(scale,2)*100+(15*i),pow(scale,2)*100+(15*i)))
        
        cv2.imwrite('OnlyBlack.png',OnlyBlack)

        _, labels = cv2.connectedComponents(OnlyBlack)
        mask = np.zeros(OnlyBlack.shape, dtype="uint8")

        # Set lower bound and upper bound criteria for characters
        total_pixels = src.shape[0] * src.shape[1]
        lower = total_pixels // 260 # heuristic param, can be fine tuned if necessary
        upper = total_pixels // 20 # heuristic param, can be fine tuned if necessary

        # Loop over the unique components
        for (i, label) in enumerate(np.unique(labels)):
            # If this is the background label, ignore it
            if label == 0:
                continue
        
            # Otherwise, construct the label mask to display only connected component
            # for the current label
            labelMask = np.zeros(OnlyBlack.shape, dtype="uint8")
            labelMask[labels == label] = 255
            
            numPixels = cv2.countNonZero(labelMask)
        
            # If the number of pixels in the component is between lower bound and upper bound, 
            # add it to our mask
            if numPixels > lower and numPixels < upper:
                mask = cv2.add(mask, labelMask)

        # Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        # Sort the bounding boxes from left to right, top to bottom
        # sort by Y first, and then sort by X if Ys are similar
        def compare(rect1, rect2):
            if abs(rect1[1] - rect2[1]) > 10:
                return rect1[1] - rect2[1]
            else:
                return rect1[0] - rect2[0]
            
        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )
        print(boundingBoxes)
        print(src.shape)
        for rect in boundingBoxes:
            x,y,w,h = rect
            h_Ori, w_Ori, c_Ori = src.shape
            
            keepWidth = w >= w_Ori/30 and w <= w_Ori/8
            keepHeight = h >= h_Ori/5 and h <= h_Ori/1.2
            numPixels = cv2.countNonZero(mask[y:y+h, x:x+w])
            density = numPixels/(h*w)
            keepDensity = density >= 0.15 and density <= 0.8 
            
            

            if keepHeight and keepWidth and keepDensity:
                crop = mask[y:y+h, x:x+w]            
                characters.append((crop,x))      
                
        cv2.imwrite('mask.png',mask)
     
        if(len(characters)>= 7):
            break
        


    if(len(characters)==0):
     
        return []
    else:
        listcharac,_ = zip(*sorted(characters,key =lambda x : x[1]))    
        return listcharac