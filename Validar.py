from random import random
import cv2
import SplitLabel
import OpencvYolo
import threading
import numpy as np
import ThreadConnections
import time
import logging
import Plate
import csv
from datetime import datetime



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
    
     

def sendToCamera(cameraConnection, characterData,cameraSemaphore,semaphoreConnections,listConnections):
    try:
        ip,port = cameraConnection.connection.getpeername()
        print("   ----Enviando foto a "+str(ip)+" con puerto "+str(port))
        
        #img = np.zeros((48,48,3))
        #img[:,:,0]=characterData.getImage()
        input_data = np.array(characterData.getImage(), dtype=np.float32)    
        
        cameraConnection.connection.send(bytes(input_data))
        cameraConnection.connection.settimeout(120)
        resultado = cameraConnection.connection.recv(30)
        caracter,confidence=str(resultado,'UTF-8').split("|")
        print("La imagen es "+caracter+" con una probabilidad de "+confidence)
         
        print("   ++++Clasificación realizada por la cámara con ip "+str(ip)+" con puerto "+str(port))
        
        characterData.isProcessing=False
        characterData.hasProcessed=True
        characterData.characterOCR=caracter
        characterData.confidence=float(confidence)
        cameraConnection.isAvailable = True
        cameraSemaphore.release()

    except:
         print ("   Error, una cámara se ha desconectado cuando se le estaba enviando una imagen")
         characterData.isProcessing=False
         characterData.hasProcessed=False

         semaphoreConnections.acquire()
         listConnections.remove(cameraConnection)
         semaphoreConnections.release()



Timestart = datetime.now()
red=OpencvYolo.Cnn('yolov4-tiny-custom.cfg','yolov4-tiny-custom_final.weights')
imgs=[]
#sockets=[]
semaphoreConnections = threading.BoundedSemaphore(value=1)
#listConnectionsConcurrent=[]
listConnections=[]
plates=[]

#El semáforo tiene el número de cámaras que hay conectadas para evitar la espera ocupada
cameraSemaphore = threading.Semaphore(value=0)


logging.basicConfig(level=logging.DEBUG)


threadConnection= ThreadConnections.ThreadConnections(semaphoreConnections,listConnections,cameraSemaphore)
threadConnection.start()

fotoName=0
matriculaReal=''
startTotalTime=0


with open('ValidacionStatsDificil2Camaras.csv', 'w', newline='') as csvfile:
    fileWriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    while True:
    
        semaphoreConnections.acquire()
        lenConnections=len(listConnections)
        semaphoreConnections.release()
        imgs.clear()

        
        if(lenConnections>0):
            '''
            #Esto simula las imagenes, cambiar por obtener las imagenes desde ip steam
            with open('config.json') as file:
                config = json.load(file)
                file.close()

            
            for camera in config['cameras']:
                #Esto lo usaré cuando tenga varias cámaras
                #imgs.append(cv2.VideoCapture("http://"+cameraIP))
                img = cv2.imread(str(camera["foto"]))
                imgs.append(img)
            '''
            
            fotoName=fotoName+1
            if(fotoName>100):
                print(datetime.now()-Timestart)

            with open('./DatasetValidacion/'+str(fotoName)+'.txt') as f:
                matriculaReal = f.read().strip()
                print('Foto :',fotoName)
            
            print(matriculaReal)
            
            
            semaphoreConnections.acquire()
            removeConnections=[]
            doOnlyOne=True
            for c in listConnections:
                try:
                    '''
                    cap = cv2.VideoCapture('http://'+str(c.ip)+':80/')
                    
                    if (not cap.isOpened() or cap is None ):
                        raise ConnectionError
                    
                    if (cap.isOpened()):
                        ret, frame = cap.read()
                    
                    if(frame is not None):
                        imgs.append(frame)
                        cv2.imshow("cameraIP: "+str(c.ip),frame)
                    cv2.waitKey(1)
                    
                    if (cap.isOpened()):
                        cap.release()'''
                        
                    if (doOnlyOne):
                        imgs.append(cv2.imread('./DatasetValidacion/'+str(fotoName)+'.jpg'))
                        doOnlyOne=False
                        
                        
                except:
                    '''if (cap.isOpened()):
                        cap.release()
                    '''
                        
                    #cerramos la conexión y bajamos el contador del semáforo en 1    
                    c.close()
                    cameraSemaphore.acquire()
                    
                    #Añadimos las conexiones que no responden para eliminarlas
                    removeConnections.append(c)
                    continue
                
            #Eliminamos las conexiones que no responden
            for rc in removeConnections:
                listConnections.remove(rc)
            
            semaphoreConnections.release()
        
            logging.debug("Imagenes a escanear: "+str(len(imgs))) 
            if(len(imgs) == 0):
                continue

            imgs=sorted(imgs,key=lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std())
            plates.clear()
            startTotalTime=time.time()
            
            for img in imgs:
                #Pasa a la red la imagen y devuelve las matriculas que vea en la imagen
                logging.debug("Búsqueda de matrícula")
                
                timestart=time.time() 
                labels=red.load_image(img)
                logging.debug("Tiempo en la búsqueda de la matrícula: "+str(time.time()-timestart)+"s | Matrículas encontradas: "+str(len(labels)))

                if (len(labels) == 0):
                    continue
            
                for label in labels:
                    cv2.imwrite("Matricula.png", label)

                    
                    listDataCharacters = [DataCharacter(cv2.resize(characImg,(48,48)), None) for characImg in SplitLabel.Split(label)]
                    
                    size = len(listDataCharacters)
                    
                    
                    '''if (not(size == 7 or size == 8)):
                        #fileWriter.writerow([str(fotoName)+'.jpg','',matriculaReal,str(size),'','','No'])
                        continue'''
                    
                    #Debug solo
                    for index, dc in enumerate(listDataCharacters):
                        cv2.imwrite(str(index)+".jpg", dc.img)

                    print("Caracteres de la matrícula",size)
                    
                    if (len(listDataCharacters) == 0):
                        #fileWriter.writerow([str(fotoName)+'.jpg','',matriculaReal,str(size),'','','No'])
                        continue

                   
                    while (any(dataCharacter.hasProcessed==False for dataCharacter in listDataCharacters)):
                    
                            cameraSemaphore.acquire()
                    
                            dataToProccess=None
                            freeConnection=None

                            #Todo esto podría estar en el any para saber si queda alguno que no  se ha procesado y quitar el if de abajo pero entonces en el caso de caída no se procesaría de nuevo
                            #Buscamos datos a procesar
                            for dc in listDataCharacters:
                                if(dc.isProcessing == False and dc.hasProcessed == False):
                                    dc.isProcessing=True
                                    dataToProccess=dc
                                    break
                    
                            #Si no hay ninguno que no esté procesandose y no se ha procesado se salta de nuevo al bucle para que compruebe si todos se han procesado, en caso de que uno no acabe lo vuelve a enviar
                            if(dataToProccess == None):
                                cameraSemaphore.release()
                                continue

                            #buscamos una conexión libre
                            semaphoreConnections.acquire()
                            for c in listConnections:
                                if(c.isAvailable == True):
                                    freeConnection = c
                                    c.isAvailable = False
                                    break;
                        
                            semaphoreConnections.release()

                            #Esto nunca se debería dar
                            if(freeConnection==None):
                                print("Se ha salido porque no se ha encontrado ninguna conexión disponible")
                                continue
                                
                        
                            hilo = threading.Thread(target=sendToCamera, args=(freeConnection,dataToProccess,cameraSemaphore,semaphoreConnections,listConnections))
                            print("EMPIEZA HILO")
                            hilo.start()

                    print("Creando matrícula")
                    matricula = Plate.Plate(listDataCharacters)
                    print("¿Es posible esta matrícula? ",matricula.plausible())
                    print("La matrícula es: ",str(matricula)," con probabilidad: ",matricula.getProability()*100,"%")
                    
                    #Solo para validar
                    #if(matricula.plausible()):
                    plates.append(matricula)
                    
                    if (matricula.getProability() > 0.98 and matricula.plausible()):
                        break
                
            if(len(plates)>0):
                
                matriculas = sorted(plates,key=lambda x:x.getProability())
                
                matricula = None
                
                for mat in matriculas:
                    if (mat.plausible()):
                        matricula = mat
                        break
                
                if(matricula == None):
                    matricula=matriculas[0]
                
                print("La matrícula más probable es: ", matricula," con probabilidad: ",matricula.getProability()*100,"%")
                aux=[]
                for dc in matricula.listDataCharacters:
                    aux.append(str(dc.characterOCR))
                    aux.append(str(dc.confidence))
                aux.append(str(time.time()-startTotalTime))
                fileWriter.writerow([str(fotoName)+'.jpg',matricula,matriculaReal,str(size),str(matricula.plausible()),str(matricula.getProability()*100),'Si'] + aux)
                
            else:
                fileWriter.writerow([str(fotoName)+'.jpg','',matriculaReal,None,'False','','No'])
                print("No se ha reconocido una matrícula coherente")


