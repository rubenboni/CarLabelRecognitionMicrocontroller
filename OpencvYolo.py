#Autor: Rubén Bonilla Tanco
import cv2 as cv
import numpy as np
import time
import sys
import pafy

class Cnn:
    ln = None
    net = None

    def __init__(self,cnn,weights):
        #creamos la red neuronal
        self.net = cv.dnn.readNetFromDarknet(cnn,weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        self.ln = self.net.getLayerNames()
        
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
    
    
    
    def load_image(self,imgInput):

        #Transformamos la imagen en blob
        ImageBlob= cv.dnn.blobFromImage(imgInput,1/255.0,(412,412),swapRB=True, crop=False)
        self.net.setInput(ImageBlob)
        outputs=self.net.forward(self.ln)

        outputs = np.vstack(outputs)

        return self.post_process(imgInput, outputs, 0.5)
    
    def post_process(self,img, outputs, conf):
        H, W = img.shape[:2]
        imgWithColor = img.copy()
        boxes = []
        confidences = []
        classIDs = []
        images=[]

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
            
                x, y, w, h = output[:4] * np.array([W, H, W, H])

                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)

                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)


                pts1= np.float32([[int(x - w//2), int(y - h//2)],[int(x + w//2), int(y - h//2)],[int(x - w//2), int(y + h//2)],[int(x + w//2), int(y + h//2)]])
                pts2= np.float32([[0,0],[480,0],[0,300],[480,300]])

                M = cv.getPerspectiveTransform(pts1,pts2)
                label = cv.warpPerspective(imgWithColor, M ,(480,300))
                images.append(label)
            

        return images



