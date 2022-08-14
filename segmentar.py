from os import listdir
import cv2
import OpencvYolo
import SplitLabel
from random import random




class DataCharacter:
    img= None
    characterOCR = None
    isProcessing=False
    hasProcessed=False
    confidence=None

    def __init__(self,_img,_character):
        self.img = _img
        self.characterOCR = _character

    def getImage(self):
        return self.img

red=OpencvYolo.Cnn('yolov4-tiny-custom.cfg','yolov4-tiny-custom_final.weights')
for f in listdir('./aux'):
    labels=red.load_image(cv2.imread('./aux/'+f))
    for label in labels:
        listDataCharacters = [DataCharacter(cv2.resize(characImg,(48,48)), None) for characImg in SplitLabel.Split(label)]
        for index, dc in enumerate(listDataCharacters):
            cv2.imwrite('./Dataset/'+str(int(random()*20000000))+".jpg", dc.img)

