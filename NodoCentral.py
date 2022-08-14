import cv2
import SplitLabel
import OpencvYolo
import threading
import numpy as np
import ThreadConnections
import time
import logging
import Plate


#Clase que representa cada caracter de una matrícula
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
    
     
#Función encargada de enviar las imágenes a las cámaras
def sendToCamera(cameraConnection, characterData,cameraSemaphore,semaphoreConnections,listConnections):
    try:
        ip,port = cameraConnection.connection.getpeername()
        print("   ----Enviando foto a "+str(ip)+" con puerto "+str(port))
    
        input_data = np.array(characterData.getImage(), dtype=np.float32)    
        
        cameraConnection.connection.send(bytes(input_data))
        cameraConnection.connection.settimeout(60)
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




red=OpencvYolo.Cnn('yolov4-tiny-custom.cfg','yolov4-tiny-custom_final.weights')
imgs=[]
semaphoreConnections = threading.BoundedSemaphore(value=1)
listConnections=[]
plates=[]

#El semáforo tiene el número de cámaras que hay conectadas para evitar la espera ocupada
cameraSemaphore = threading.Semaphore(value=0)


logging.basicConfig(level=logging.DEBUG)


threadConnection= ThreadConnections.ThreadConnections(semaphoreConnections,listConnections,cameraSemaphore)
threadConnection.start()


while True:
   
    semaphoreConnections.acquire()
    lenConnections=len(listConnections)
    semaphoreConnections.release()
    imgs.clear()

    
    if(lenConnections>0):
        semaphoreConnections.acquire()
        removeConnections=[]
        #Obtenemos las imágenes de las cámaras
        for c in listConnections:
            try:
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
                    cap.release()
                    
            except:
                if (cap.isOpened()):
                    cap.release()
                    
                #Cerramos la conexión y bajamos el contador del semáforo en 1    
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
        
        for img in imgs:
            #Pasa a la red la imagen y devuelve las matriculas que vea en la imagen
            logging.debug("Búsqueda de matrícula")
            
            timestart=time.time() 
            labels=red.load_image(img)
            logging.debug("Tiempo en la búsqueda de la matrícula: "+str(time.time()-timestart)+"s | Matrículas encontradas: "+str(len(labels)))

            if (len(labels) == 0):
                continue


            #Segmentación en caracteres la matrícula detectada
            #Siempre se supone que solamente hay una matrícula en la imagen
            listDataCharacters = [DataCharacter(cv2.resize(characImg,(48,48)), None) for characImg in SplitLabel.Split(labels[0])]
            
            size = len(listDataCharacters)
            
            #Si no tiene un mínimo de caracteres, se descarta
            if (not(size == 7 or size == 8)):
                logging.debug("Longitud de la matrícula, no válida")
                continue

            logging.debug("Caracteres de la matrícula",size)
            
            #Si no hay ninguna cámara no continúa
            if (len(listDataCharacters) == 0):
                continue

            #Mientras que no se hayan procesado todos los caracteres
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

                    #Se busca una conexión libre
                    semaphoreConnections.acquire()
                    for c in listConnections:
                        if(c.isAvailable == True):
                            freeConnection = c
                            c.isAvailable = False
                            break;
                    semaphoreConnections.release()

                    
                    if(freeConnection==None):
                        logging.debug("No hay ninguna conexión disponible")
                        continue
                        
                
                    hilo = threading.Thread(target=sendToCamera, args=(freeConnection,dataToProccess,cameraSemaphore,semaphoreConnections,listConnections))
                    logging.debug("Nuevo hilo creado para enviar imágen a una cámara")
                    hilo.start()

            logging.debug("Datos de la matrícula según una cámara:")
            matricula = Plate.Plate(listDataCharacters)
            logging.debug("¿Es posible esta matrícula? ",matricula.plausible())
            logging.debug("La matrícula es: ",str(matricula)," con probabilidad: ",matricula.getProability()*100,"%")
            
            if(matricula.plausible()):
                plates.append(matricula)
            
            if (matricula.getProability() > 0.98 and matricula.plausible()):
                break
        
        print("Datos de la matrícula :")  
        if(len(plates)>0):
            
            matricula = sorted(plates,key=lambda x:x.getProability())[0]
            print("La matrícula más probable es: ", matricula," con probabilidad: ",matricula.getProability()*100,"%")
            
        else:
            
            print("No se ha reconocido una matrícula coherente")


