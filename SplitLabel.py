

from os import wait
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def Split(src):
    h, w, c = src.shape
    characters = []



    mask = cv2.inRange(src,np.array([0,0,0]),np.array([100,100,100])) 

    
    cv2.imwrite('mask.png',mask)
    


    '''gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    
    cv2.imwrite("gray.png",gray)
    
    img = cv2.medianBlur(gray,5)'''
    
    #_, thresh = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    
    
    #thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    
    #cv2.imwrite("thresh.png",src)
    
    output = cv2.connectedComponentsWithStats(mask,cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(mask.shape,dtype="uint8")
    
    for i in range(1,numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]       
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        
        # ensure the width, height, and area are all neither too small
    	# nor too big       
        keepWidth = w > 10 and w < 80
        keepHeight = h > 80 and h < 220
        keepArea = area > 1000 and area < 9000
        
        '''
        keepWidth = w > 10 and w < 80
        keepHeight = h > 80 and h < 220
        keepArea = area > 1000 and area < 10000
        '''
        (cX, cY) = centroids[i]

    	# ensure the connected component we are examining passes all
    	# three tests
        if all((keepWidth, keepHeight,keepArea)):
    		# construct a mask for the current connected component and
    		# then take the bitwise OR with the mask
            componentMask = (labels == i).astype("uint8") * 255

            #cv2.putText(componentMask,"h:"+str(h)+" w:"+str(w)+" area:"+str(area),(20,60), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)
            mask = cv2.bitwise_or(mask, componentMask)
            character=componentMask[y:y+h,x:x+w]
            characters.append((character,x))
            

    if(len(characters)==0):
        return []
    else:
        listcharac,_ = zip(*sorted(characters,key =lambda x : x[1]))    
        return listcharac